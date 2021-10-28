// modified from https://github.com/NVIDIA/apex/blob/master/csrc/compat.h
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#ifdef VERSION_GE_1_3
#define DATA_PTR data_ptr
#else
#define DATA_PTR data
#endif