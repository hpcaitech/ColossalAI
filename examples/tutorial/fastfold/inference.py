# Copyright 2023 HPC-AI Tech Inc.
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time

import fastfold
import numpy as np
import torch
import torch.multiprocessing as mp
from fastfold.config import model_config
from fastfold.data import data_transforms
from fastfold.model.fastnn import set_chunk_size
from fastfold.model.hub import AlphaFold
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.tensor_utils import tensor_tree_map

if int(torch.__version__.split(".")[0]) >= 1 and int(torch.__version__.split(".")[1]) > 11:
    torch.backends.cuda.matmul.allow_tf32 = True


def random_template_feats(n_templ, n):
    b = []
    batch = {
        "template_mask": np.random.randint(0, 2, (*b, n_templ)),
        "template_pseudo_beta_mask": np.random.randint(0, 2, (*b, n_templ, n)),
        "template_pseudo_beta": np.random.rand(*b, n_templ, n, 3),
        "template_aatype": np.random.randint(0, 22, (*b, n_templ, n)),
        "template_all_atom_mask": np.random.randint(0, 2, (*b, n_templ, n, 37)),
        "template_all_atom_positions": np.random.rand(*b, n_templ, n, 37, 3) * 10,
        "template_torsion_angles_sin_cos": np.random.rand(*b, n_templ, n, 7, 2),
        "template_alt_torsion_angles_sin_cos": np.random.rand(*b, n_templ, n, 7, 2),
        "template_torsion_angles_mask": np.random.rand(*b, n_templ, n, 7),
    }
    batch = {k: v.astype(np.float32) for k, v in batch.items()}
    batch["template_aatype"] = batch["template_aatype"].astype(np.int64)
    return batch


def random_extra_msa_feats(n_extra, n):
    b = []
    batch = {
        "extra_msa": np.random.randint(0, 22, (*b, n_extra, n)).astype(np.int64),
        "extra_has_deletion": np.random.randint(0, 2, (*b, n_extra, n)).astype(np.float32),
        "extra_deletion_value": np.random.rand(*b, n_extra, n).astype(np.float32),
        "extra_msa_mask": np.random.randint(0, 2, (*b, n_extra, n)).astype(np.float32),
    }
    return batch


def generate_batch(n_res):
    batch = {}
    tf = torch.randint(21, size=(n_res,))
    batch["target_feat"] = torch.nn.functional.one_hot(tf, 22).float()
    batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
    batch["residue_index"] = torch.arange(n_res)
    batch["msa_feat"] = torch.rand((128, n_res, 49))
    t_feats = random_template_feats(4, n_res)
    batch.update({k: torch.tensor(v) for k, v in t_feats.items()})
    extra_feats = random_extra_msa_feats(5120, n_res)
    batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})
    batch["msa_mask"] = torch.randint(low=0, high=2, size=(128, n_res)).float()
    batch["seq_mask"] = torch.randint(low=0, high=2, size=(n_res,)).float()
    batch.update(data_transforms.make_atom14_masks(batch))
    batch["no_recycling_iters"] = torch.tensor(2.)

    add_recycling_dims = lambda t: (t.unsqueeze(-1).expand(*t.shape, 3))
    batch = tensor_tree_map(add_recycling_dims, batch)

    return batch


def inference_model(rank, world_size, result_q, batch, args):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()
    torch.cuda.set_device(rank)
    config = model_config(args.model_name)
    if args.chunk_size:
        config.globals.chunk_size = args.chunk_size

    config.globals.inplace = args.inplace
    config.globals.is_multimer = False
    model = AlphaFold(config)

    model = inject_fastnn(model)
    model = model.eval()
    model = model.cuda()

    set_chunk_size(model.globals.chunk_size)

    with torch.no_grad():
        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
        t = time.perf_counter()
        out = model(batch)
        print(f"Inference time: {time.perf_counter() - t}")
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    result_q.put(out)

    torch.distributed.barrier()
    torch.cuda.synchronize()


def inference_monomer_model(args):
    batch = generate_batch(args.n_res)
    manager = mp.Manager()
    result_q = manager.Queue()
    torch.multiprocessing.spawn(inference_model, nprocs=args.gpus, args=(args.gpus, result_q, batch, args))
    out = result_q.get()

    # get unrelexed pdb and save
    # batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
    # plddt = out["plddt"]
    # plddt_b_factors = np.repeat(plddt[..., None], residue_constants.atom_type_num, axis=-1)
    # unrelaxed_protein = protein.from_prediction(features=batch,
    #                                             result=out,
    #                                             b_factors=plddt_b_factors)
    # with open('demo_unrelex.pdb', 'w+') as fp:
    #     fp.write(unrelaxed_protein)


def main(args):
    inference_monomer_model(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="""Number of GPUs with which to run inference""")
    parser.add_argument("--n_res", type=int, default=50, help="virtual residue number of random data")
    parser.add_argument("--model_name", type=str, default="model_1", help="model name of alphafold")
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--inplace', default=False, action='store_true')

    args = parser.parse_args()

    main(args)
