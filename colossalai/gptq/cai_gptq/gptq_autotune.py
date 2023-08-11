import math
import time
import torch

class AutoTune:
    def __init__(self, tune_func, warmup=10, bech_run=20):

        self.func = tune_func
        self.warmup_num = warmup
        self.bech_run = bech_run
        self.config_caches = {}

    def prune_configs(self, tune_config):

        norm_configs = []
        linear_configs = []

        if tune_config['qkv_fused']:
            max_in = 2**int(math.log2(tune_config['in_dim']))
            in_dim = tune_config['in_dim']
        else:
            max_in = 2**int(math.log2(tune_config['in_dim']))
            in_dim = tune_config['in_dim']

        max_out = 2**int(math.log2(tune_config['out_dim']))
        if max_out > 1024:
            max_out = 1024
        m = 2
        n = 64
        x = 64
        y = 1

        # while n <= max_out:
        #     ret_config = {
        #         "linear_x": n,
        #         "linear_y": in_dim,
        #     }
        #     linear_configs.append(ret_config)
        #     n = n * 2
        while n <= max_out:
            m = 64
            while m < in_dim:
                ret_config = {
                    "linear_x": n,
                    "linear_y": m,
                }
                linear_configs.append(ret_config)
                m = m * 2
            ret_config = {
                "linear_x": n,
                "linear_y": in_dim,
            }
            linear_configs.append(ret_config)
            n = n * 2

        # if tune_config['act_type'] == 0:
        #     while n <= max_out:
        #         m = 64
        #         while m <= max_in:
        #             ret_config = {
        #                 "norm_x": 0,
        #                 "norm_y": 0,
        #                 "linear_x": n,
        #                 "linear_y": m,
        #             }
        #             linear_configs.append(ret_config)
        #             m = m * 2
        #         n = n * 2
        # elif tune_config['act_type'] > 0:
        #     while n <= max_out:
        #         ret_config = {
        #             "norm_x": 0,
        #             "norm_y": 0,
        #             "linear_x": n,
        #             "linear_y": in_dim,
        #         }
        #         linear_configs.append(ret_config)
        #         n = n * 2            
        return linear_configs

    def warmup(self, tune_config, *args, **kwargs):
        # if tune_config['qkv_fused']:
        #     output = torch.zeros(3, tune_config['input_len'], tune_config['out_dim'],  
        #         dtype = torch.float16, device=torch.cuda.current_device()).contiguous()
        # else:
        #     output = torch.zeros(tune_config['input_len'], tune_config['out_dim'], 
        #         dtype = torch.float16, device=torch.cuda.current_device()).contiguous()

        for i in range(0, self.warmup_num):
            # out = self.func(*args[:6], output, *args[7:],**kwargs)
            out = self.func(*args, **kwargs)

    def benchmark(self, tune_config, *args, **kwargs):

        self.warmup(tune_config, *args, **kwargs)
        linear_configs = self.prune_configs(tune_config)
        # print(ret_configs)
        times = {}
        best_norm_x = 512
        best_norm_y = 1
        best_linear_x = 512
        best_linear_y = 256
        # if best_linear_x > tune_config['out_dim']:
        #     best_linear_x = 2**int(math.log2(tune_config['out_dim']))
        # if best_linear_y > tune_config['in_dim'] and tune_config['qkv_fused'] == False:
        #     best_linear_y = 2**int(math.log2(tune_config['in_dim']))    
        # if best_linear_y > tune_config['in_dim'] and tune_config['qkv_fused']:
        #     best_linear_y = 2**int(math.log2(tune_config['in_dim'] //3))    
        if best_norm_x > tune_config['input_dim']:
            best_norm_x = 2**int(math.log2(tune_config['input_dim']))

        if tune_config['wdtype'] == torch.int8:
            nweights = 2
        elif tune_config['wdtype'] == torch.int32:
            nweights = 8
        elif tune_config['wdtype'] == torch.int64:
            nweights = 16
        times = {}

        # if tune_config['qkv_fused']:
        #     output = torch.zeros(3, tune_config['input_len'], tune_config['out_dim'],  
        #         dtype = torch.float16, device=torch.cuda.current_device()).contiguous()
        # else:
        #     output = torch.zeros(tune_config['input_len'], tune_config['out_dim'], 
        #         dtype = torch.float16, device=torch.cuda.current_device()).contiguous()

        for config in linear_configs:
            linear_x = config['linear_x']
            linear_y = config['linear_y']
            print(config)
            start = time.time()
            for run in range(0, self.bech_run):
                # out = self.func(*args[:6], output, *args[7:-2], linear_x, linear_y)
                out = self.func(*args[:-2], linear_x, linear_y)

            torch.cuda.synchronize()
            end = time.time()              
            times[' '.join(map(str,config.values()))] = end - start
            # print(f"{config}: {end-start:.6f}")
        sorted_dict = sorted(times.items(), key=lambda x:x[1])
        values = sorted_dict[0][0].split()
        # print(sorted_dict)
        best_linear_x = int(values[0])
        best_linear_y = int(values[1])

        times = {}

        key = ' '.join(map(str,tune_config.values()))
        
        ret_config = {
            "linear_x": best_linear_x,
            "linear_y": best_linear_y,
        }
        self.config_caches[key] = ret_config
        # print("best config:", tune_config, ret_config)
    def get_best_config(self, tune_config, *args, **kwargs):
        key = ' '.join(map(str,tune_config.values()))

        if key in self.config_caches:
            return self.config_caches[key]
        else:
            # print(tune_config)
            self.benchmark(tune_config, *args, **kwargs)
            return self.config_caches[key]


                
