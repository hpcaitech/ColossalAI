import torch
from torch.fx import GraphModule
from colossalai.fx.passes.adding_split_node_pass import split_with_split_nodes_pass, balanced_split_pass
from colossalai.fx import ColoTracer
import random
import numpy as np

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

def split_model_and_get_DAG(model, data_gen):
    model.eval()

    # generate input sample
    kwargs = data_gen()

    # get origin output and rng state
    cpu_rng_state = torch.get_rng_state()
    output = model(**kwargs)

    # tracing model
    tracer = ColoTracer()
    try:
        meta_args = {k: v.to('meta') for k, v in kwargs.items()}
        graph = tracer.trace(root=model, meta_args=meta_args)
    except Exception as e:
        raise RuntimeError(f"Failed to trace {model.__class__.__name__}, error: {e}")
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    # apply transform passes
    annotated_model = balanced_split_pass(gm, 2)
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model)

    return top_module, split_submodules[0]._DAG

def check_input(input, input_node, top_module):
    for user in input_node.users.keys():
        partition_name = user.name
        assert partition_name in input['output']
        
def check_submod(submod_partition, node, top_module):
    for arg in node.args:
        input_part_name = None
        if arg.op == 'placeholder':
            input_part_name = 'MODEL_INPUT'
        elif not arg.name.startswith('getitem'):
            input_part_name = arg.name
        else:
            input_part_name = arg.args[0].name
        assert input_part_name in submod_partition['input']
        
    for user in node.users:
        output_part_names = []
        if user.op == 'output':
            output_part_names.append('MODEL_OUTPUT')
        elif not user.name.startswith('getitem'):
            output_part_names.append(user.name)
        else:
            for n in user.users:
                if n.op == 'output':
                    output_part_names.append('MODEL_OUTPUT')
                else:
                    output_part_names.append(n.name)
            
        for output_part_name in output_part_names:
            assert output_part_name in submod_partition['output']

def check_DAG(top_module, DAG):    
    assert 'input_partition' in DAG
    input_partition = DAG['input_partition']
    
    for node in top_module.graph.nodes:
        # check input
        if node.op == 'placeholder':
            assert node.name in input_partition
            input = input_partition[node.name]
            check_input(input, node, top_module)
        elif node.op == 'call_module':
            assert node.name in DAG
            submod_partition = DAG[node.name]
            check_submod(submod_partition, node, top_module)
            