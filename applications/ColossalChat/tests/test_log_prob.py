import torch
import time
import random

def code1(target, vocab_start_index, vocab_end_index):
    """index Put"""
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0
    return masked_target

def code2(target, vocab_start_index, vocab_end_index):
    """bool multiply"""
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target *= ~target_mask
    return masked_target

def test_performance():
    batch_size = 8
    sizes = [4096, 8192, 16384, 32768, 131072]
    code1_times = []
    code2_times = []
    
    for size in sizes:
        target = torch.randint(0, size, (batch_size, size,)).to("npu")
        vocab_start_index = random.randint(0, size//2)
        vocab_end_index = random.randint(size//2, size)
        
        # warmup
        for _ in range(5):
            code1(target, vocab_start_index, vocab_end_index)
            code2(target, vocab_start_index, vocab_end_index)
        
        # Code 1: index input
        start_time = time.time()
        for _ in range(10):
            code1(target, vocab_start_index, vocab_end_index)
        code1_time = (time.time() - start_time) / 10
        code1_times.append(code1_time)
        
        # Code 2: bool multiply
        start_time = time.time()
        for _ in range(10):
            code2(target, vocab_start_index, vocab_end_index)
        code2_time = (time.time() - start_time) / 10
        code2_times.append(code2_time)
        
        print(f"DataSize: {size}")
        print(f"  Code 1:index input AvgRuntime: {code1_time:.6f} s")
        print(f"  Code 2:bool multiply AvgRuntime {code2_time:.6f} s")
        # print(f"  acceleration ratio: {(code1_time/code2_time-1)*100:.2f}%")
        print(f"  acceleration ratio: {(code1_time/code2_time - 1)*100:.2f}%")


if __name__ == "__main__":    
    print("\n===== Performance Benchmark =====")
    test_performance()