#!/bin/bash

python -m torch.distributed.launch test_2d.py --nproc_per_node 8 test_3d.py --host $HOST --port 29516 --world_size 8

# expected test output
#  distributed environment initialized
#  AB forward: pass
#  AB backward: pass
#  ABT forward: pass
#  ABT backward: pass
#  ATB forward: pass
#  ATB backward: pass
#  linear backward: pass
#  linear backward: pass
#  layer norm forward: pass
#  layer norm backward: pass
#  self attention forward: pass
#  self attention backward: pass
#  mlp forward: pass
#  mlp backward: pass
#  transformerlayer forward: pass
#  transformerlayer backward: pass