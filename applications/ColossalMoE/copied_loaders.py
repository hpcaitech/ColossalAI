from colossalai.tensor.moe_tensor.api import get_dp_rank, get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor

    # Copied from colossalai.moe
    def pre_save_model(self, model: nn.Module) -> dict:
        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            if ".experts." in name and is_moe_tensor(param):
                ep_group = param.ep_group
                ep_rank = dist.get_rank(ep_group)
                ep_size = dist.get_world_size(ep_group)
                dp_rank = get_dp_rank(param)
                if dp_rank == 0:
                    param = param.data.cuda()
                    if ep_rank == 0:
                        all_param = [torch.zeros_like(param) for _ in range(ep_size)] 
                    else:
                        all_param = None
                    # gather param from every ep rank
                    # TODO: Switch to gather
                    # dist.all_gather(all_param, param, group=ep_group)
                    dist.gather(param, all_param, group=ep_group)
                    if ep_rank == 0:
                        all_param = torch.cat(all_param, dim=0)
                        state_dict[name] = all_param.cpu()
        if self.pp_size > 1:
            if self.dp_rank == 0:
                out = [None for _ in range(self.pp_size)]
                # dist.all_gather_object(out, state_dict, group=self.pp_group)
                dist.gather_object(state_dict, out, group=self.pp_group)
                if self.pp_rank == 0:
                    new_state_dict = {}
                    for o in out:
                        new_state_dict.update(o)
                    state_dict = new_state_dict
        dist.barrier()
        return state_dict

    def save_unsharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        gather_dtensor: bool,
        use_safetensors: bool,
    ):
        state_dict = self.pre_save_model(model)
        if dist.get_rank() == 0:
            torch.save(state_dict, checkpoint)
        dist.barrier()

    # Copied from colossalai.moe
    def save_unsharded_optimizer(self, optimizer: OptimizerWrapper, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer state dict to a file with given path.

        Args:
            optimizer (OptimizerWrapper): Optimizer to save sharded state_dict.
            checkpoint (str): Path to save optimizer state_dict.
            gather_dtensor (bool): Whether to gather_dtensor, not used.
        """
        if self.coordinator.is_master():
            logging.warning("Please avoid using unsharded checkpointing methods when dealing with large models!")

        assert isinstance(optimizer, OptimizerWrapper), "Please boost the optimizer before saving!"

        # optimizer states of parameters kept by local device('s pipeline stage)
        local_states = dict()

        for param, state in optimizer.optim.state.items():
            if param is None:
                continue

            # working param is needed for obtaining correct param_id
            master_to_working_map = optimizer.get_master_to_working_map()
            if master_to_working_map is not None and id(param) in master_to_working_map:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param

            # gather complete state from tp shards & dp shards
            param_id = optimizer.param_info["param2id"][id(working_param)]
            local_states[param_id] = self.pre_save_optim(
                state,
                working_param,
                inplace=False,
                device=torch.device("cuda"),
            )

        if self.pp_size == 1:
            # When pipeline is not used, let master rank directly save the collected state_dict.
            state_dict = {"param_groups": optimizer.optim.param_groups, "state": local_states}
            if self.coordinator.is_master():
                save_state_dict(state_dict, checkpoint, use_safetensors=False)
        else:
            # When pipeline is used, first collect state_dict from every pipeline stage, then save the complete state_dict.
            states_list = [None for _ in range(self.pp_size)]
            dist.barrier(self.pp_group)
            # dist.all_gather_object(states_list, local_states, self.pp_group)
            dist.gather_object(local_states, states_list, self.pp_group)

            # Only the master rank do the saving.
            if self.coordinator.is_master():
                state_dict = {"param_groups": optimizer.optim.param_groups, "state": dict()}
                for _states in states_list:
                    state_dict["state"].update(_states)
                save_state_dict(state_dict, checkpoint, use_safetensors=False)
        dist.barrier()

    # Copied from colossalai.moe
    def load_unsharded_optimizer(self, optimizer: OptimizerWrapper, checkpoint: str, strict: bool = False):
        """
        Load optimizer from a file with given path.

        Args:
            optimizer (OptimizerWrapper): The optimizer to be loaded.
            checkpoint_index_file (str): Path to the checkpoint file.
        """

        def _get_param_id_from_optimizer_param(
            param: torch.Tensor, master_to_working_map: Optional[Dict[int, torch.Tensor]] = None
        ):
            if master_to_working_map is not None and id(param) in master_to_working_map:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param
            if id(working_param) in optimizer.param_info["param2id"]:
                return optimizer.param_info["param2id"][id(working_param)]
            else:
                None

        if self.coordinator.is_master():
            logging.warning("Please avoid using unsharded checkpointing methods when dealing with large models!")

        assert isinstance(optimizer, OptimizerWrapper), "Please boost the optimizer before loading!"

        # Complete optimizer state_dict loaded from checkpoint, need to be processed later.
        state_dict = load_state_dict(checkpoint)

        # Load param_groups.
        updated_groups = []
        saved_groups = state_dict["param_groups"]
        for old_pg, saved_pg in zip(optimizer.optim.param_groups, saved_groups):
            new_pg = copy.deepcopy(saved_pg)
            new_pg["params"] = old_pg["params"]  # Only keep the parameters kept by current pipeline stage.
            updated_groups.append(new_pg)

        # ep extra group
        # if MOE_MANAGER.parallel == "EP":
        if self.ep_size > 1:
            new_pg = copy.deepcopy(saved_pg)
            new_pg["params"] = optimizer.optim.param_groups[-1][
                "params"
            ]  # Only keep the parameters kept by current pipeline stage.
            for param in new_pg["params"]:
                param.data = param.data.to(torch.float32)
            updated_groups.append(new_pg)
        optimizer.optim.__dict__.update({"param_groups": updated_groups})

        # Load saved states to optimizer. First discard those states not belonging to current pipeline stage.
        master_to_working_map = optimizer.get_master_to_working_map()
        id_map = {}
        for pg in optimizer.optim.param_groups:
            for param in pg["params"]:
                param_id = _get_param_id_from_optimizer_param(param, master_to_working_map)
                if param_id is not None:
                    id_map[param_id] = param
        load_states_into_optimizer(optimizer.optim, state_dict["state"], id_map, strict=True)

        # Then shard the loaded optimizer states if using tp/zero.
        for param, state in optimizer.optim.state.items():
            if param is None:
                continue
            device = param.device
            if master_to_working_map is not None and id(param) in master_to_working_map:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param
            original_shape = optimizer.param_info["param2shape"][id(working_param)]
            sharded_state = self.pre_load_optim(
                state,
                param,
                current_shape=working_param.shape,
                original_shape=original_shape,
                device=device,
                inplace=True,
            )
            optimizer.optim.state[param] = sharded_state
        sharded_optimizer_loading_epilogue(optimizer.optim)
        dist.barrier()
