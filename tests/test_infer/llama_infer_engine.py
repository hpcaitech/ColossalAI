
import torch.distributed as dist
import torch
import torch.nn as nn
from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.inference import MemoryManager
from colossalai.shardformer.policies.llama import LlamaModelInferPolicy, LlamaPolicy
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
from torch.profiler import profile, record_function, ProfilerActivity

GIGABYTE = 1024 ** 3
torch.backends.cudnn.enabled = True
DP_DIM, PP_DIM, TP_DIM = 0, 1, 2

def print_device_memory():
     if torch.cuda.is_available():
         current_device = torch.cuda.current_device()
         print(f"Currently using GPU: {current_device}")

         # free memory and the total available memory in bytes
         global_free_memory, total_GPU_memory_occupied = torch.cuda.mem_get_info()
         memory_allocated = torch.cuda.memory_allocated()
         max_memory_allocated = torch.cuda.max_memory_allocated()
         memory_reserved = torch.cuda.memory_reserved()
         max_memory_reserved = torch.cuda.max_memory_reserved()

         print(
             f"  free memory : {global_free_memory / GIGABYTE:.4f} GB,\n"
             f"  total memory: {total_GPU_memory_occupied / GIGABYTE:.4f} GB,\n"
             f"  memory allocated: {memory_allocated / GIGABYTE:.4f} GB,\n"
             f"  Max CUDA memory allocated: {max_memory_allocated / GIGABYTE:.4f} GB,\n"
             f"  memory reserved/cached: {memory_reserved / GIGABYTE:.4f} GB,\n"
             f"  Max CUDA memory reserved/cached: {max_memory_reserved / GIGABYTE:.4f} GB,\n"
         )

def print_perf_stats(latency_set, config, bs, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2  # float16

        print(f"num_layers: {num_layers}, hidden_size: {config.hidden_size}")

        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * bs / 1e12))
        print("Avg Throughput: tokens/s: {}".format((1000/(avg * 1000))))
        
def init_to_get_rotary(self, base=10000):
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0
    if hasattr(self.config,"max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config,"max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len =  2048 * rope_scaling_factor
    base = float(base)
    inv_freq = 1.0 / (base ** (torch.arange(0, self.config.head_dim_, 2, device="cpu", dtype=torch.float32) / self.config.head_dim_))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
    return

class TPCacheManagerInferenceEngine:
    
    def __init__(
        self,
        input_len: int,
        output_len: int,
        bs: int,
        tp_size: int,
    ) -> None:
        self.pg_mesh = ProcessGroupMesh(1, 1, tp_size)
        self.input_len = input_len
        self.output_len = output_len
        self.bs = bs
        self.tp_size = tp_size
        
    def init_and_insert_cache_manager(self):
        
        max_total_token_num = self.bs * (self.input_len + self.output_len)
        
        head_num = self.model.config.num_attention_heads // self.tp_size
        
        self.cache_manager = MemoryManager(
                        max_total_token_num,
                        torch.float16,
                        head_num,
                        self.model.config.hidden_size // self.model.config.num_attention_heads, 
                        self.model.config.num_hidden_layers,
                        device="cuda",
                    )
        
        setattr(self.model.model, 'cache_manager', self.cache_manager)
        
        block_loc = torch.empty(self.bs, self.input_len + self.output_len, dtype=torch.long, device="cuda")
        start_loc = torch.zeros(self.bs, dtype=torch.int32, device="cuda")
        seq_len = torch.zeros(self.bs, dtype=torch.int32, device="cuda")
        max_total_token_num = self.bs * (self.input_len)
        max_len_in_batch = self.input_len 
        for i in range(self.bs):
            block_loc[i, 0:self.input_len] = i * self.input_len + torch.arange(0, self.input_len, dtype=torch.int32, device="cuda")
            start_loc[i] = i * self.input_len
            seq_len[i] = self.input_len
        setattr(self.model.model, 'block_loc', block_loc)
        setattr(self.model.model, 'start_loc', start_loc)
        setattr(self.model.model, 'seq_len', seq_len)
        setattr(self.model.model,'total_token_num', max_total_token_num)
        setattr(self.model.model,'max_len_in_batch',max_len_in_batch)

        
    def prepare_model(self):
        llama_model_path = "/data/scratch/llama-7b-hf"
        tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        self.model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tokenizer.eos_token_id)
        init_to_get_rotary(self.model.model, base=10000)
        self.model = self.model.half()
        self.model.to(torch.cuda.current_device())
        

    def build_model(self):
        # create new model
        self.orgin_model = self.model
        shardconfig = ShardConfig(
            tensor_parallel_process_group=self.pg_mesh.get_group_along_axis(TP_DIM),
            enable_tensor_parallelism=True,
            inference_only=True,
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        
        if self.bs >= 4:
            self.init_and_insert_cache_manager()
            policy = LlamaModelInferPolicy()
        else:
            policy = LlamaPolicy()
        
        self.model, _ = shardformer.optimize(self.model, policy)
    
    def generate_data(self):
        self.input_tokens={"input_ids":torch.randint(1, 1000, (self.bs, self.input_len))}
        for t in self.input_tokens:
            if torch.is_tensor(self.input_tokens[t]):
                self.input_tokens[t] = self.input_tokens[t].to(torch.cuda.current_device())
                print(f" input_tokens[{t}].shape: {self.input_tokens[t].shape}")
                self.input_len = self.input_tokens[t].shape[1]
                print(f" input_len: {self.input_len}")
            
    def run_infer(self, test_origin=True):
        
        if test_origin:
            model = self.orgin_model
        else:
            model = self.model
        
        generate_kwargs = dict(max_new_tokens=self.output_len, do_sample=False)
        
        iters = 10
        times = []
        outputs_list = []
        warmup = 3
        for i in range(iters):
            torch.cuda.synchronize()
            start = time.time()
            outputs = model.generate(**self.input_tokens, 
                    **generate_kwargs, early_stopping=False)
            outputs_list.append(outputs)
            torch.cuda.synchronize()

            end = time.time()
            num_tokens_generation = outputs.shape[1] - self.input_len
            print(f"num_tokens_generation: {num_tokens_generation}")
            print(f"generation time is {(end - start) * 1000} ms")
            time_spend = (end-start)/num_tokens_generation
            times.append(time_spend)
            
            if test_origin:
                model.model.cache_manager.free_all()
                block_loc = torch.empty(self.bs, self.input_len + self.output_len, dtype=torch.long, device="cuda")
                start_loc = torch.zeros(self.bs, dtype=torch.int32, device="cuda")
                seq_len = torch.zeros(self.bs, dtype=torch.int32, device="cuda")
                max_len_in_batch = self.input_len 
                for i in range(self.bs):
                    block_loc[i, 0:self.input_len] = i * self.input_len + torch.arange(0, self.input_len, dtype=torch.int32, device="cuda")
                    start_loc[i] = i * self.input_len
                    seq_len[i] = self.input_len
        
                setattr(model.model, 'block_loc', block_loc)
                setattr(model.model, 'start_loc', start_loc)
                setattr(model.model, 'seq_len', seq_len)
                setattr(model.model,'max_len_in_batch',max_len_in_batch)

        print_device_memory()
        total_time = (end - start) * 1000
        token_latency = total_time/(self.output_len)
        print(outputs.shape)
        print('per_token_latency',token_latency,'ms')
        print_perf_stats(times, self.model.config, self.bs, warmup=warmup)
        
        return outputs_list
        

        