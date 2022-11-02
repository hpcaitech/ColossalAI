from get_one_alpha_beta import execute_cmd


def get_alpha_beta(width, height):
  device_list = [4,5]
  omp_prefix = "OMP_NUM_THREADS=1"
  cuda_prefix = "CUDA_VISIBLE_DEVICES=4,5"
  python_prefix = f"python -m torch.distributed.run --nproc_per_node={len(device_list)} --master_port 11000 get_one_alpha_beta.py"
  execute_cmd([omp_prefix, cuda_prefix, python_prefix])

if __name__ == "__main__":
  get_alpha_beta(1,1)