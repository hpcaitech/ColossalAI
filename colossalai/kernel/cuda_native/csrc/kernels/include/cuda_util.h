#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <math_constants.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
void check_gpu_error(T result, char const *const func, const char *const file,
                     int const line);

#define CHECK_GPU_ERROR(val) check_gpu_error((val), #val, __FILE__, __LINE__)

template <typename T>
void print_vec(const T *outv, std::string outn, int num_output_ele);

template <typename T>
T *cuda_malloc(size_t ele_num);

void cuda_free(void *pdata);

template <typename T>
void check_nan_inf(const T *data_ptr, int dsize, bool check_nan_inf,
                   std::string file, int line, cudaStream_t stream);

#define CHECK_NAN_INF(ptr, size, stream)                            \
  check_nan_inf((ptr), (size), true, __FILE__, __LINE__, (stream)); \
  check_nan_inf((ptr), (size), false, __FILE__, __LINE__, (stream))
