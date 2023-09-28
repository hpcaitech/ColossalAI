# Process OPT-175B weights

You should download the pre-trained weights following the [doc](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT) before reading this.

First, install `metaseq` and `git clone https://github.com/facebookresearch/metaseq.git`.

Then, `cd metaseq`.

To consolidate checkpoints to eliminate FSDP:

```shell
bash metaseq/scripts/reshard_mp_launch_no_slurm.sh <directory_where_all_the_shards_are>/checkpoint_last <output_dir>/ 8 1
```

You will get 8 files in `<output_dir>`, and you should have the following checksums:
```
7e71cb65c4be784aa0b2889ac6039ee8  reshard-model_part-0-shard0.pt
c8123da04f2c25a9026ea3224d5d5022  reshard-model_part-1-shard0.pt
45e5d10896382e5bc4a7064fcafd2b1e  reshard-model_part-2-shard0.pt
abb7296c4d2fc17420b84ca74fc3ce64  reshard-model_part-3-shard0.pt
05dcc7ac6046f4d3f90b3d1068e6da15  reshard-model_part-4-shard0.pt
d24dd334019060ce1ee7e625fcf6b4bd  reshard-model_part-5-shard0.pt
fb1615ce0bbe89cc717f3e5079ee2655  reshard-model_part-6-shard0.pt
2f3124432d2dbc6aebfca06be4b791c2  reshard-model_part-7-shard0.pt
```

Copy `flat-meta.json` to `<output_dir>`.

Then cd to this dir, and we unflatten parameters.

```shell
bash unflat.sh <output_dir>/ <new_output_dir>/
```

Finally, you will get 8 files in `<new_output_dir>` with following checksums:
```
6169c59d014be95553c89ec01b8abb62  reshard-model_part-0.pt
58868105da3d74a528a548fdb3a8cff6  reshard-model_part-1.pt
69b255dc5a49d0eba9e4b60432cda90b  reshard-model_part-2.pt
002c052461ff9ffb0cdac3d5906f41f2  reshard-model_part-3.pt
6d57f72909320d511ffd5f1c668b2beb  reshard-model_part-4.pt
93c8c4041cdc0c7907cc7afcf15cec2a  reshard-model_part-5.pt
5d63b8750d827a1aa7c8ae5b02a3a2ca  reshard-model_part-6.pt
f888bd41e009096804fe9a4b48c7ffe8  reshard-model_part-7.pt
```
