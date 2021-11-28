#pragma once

#include <cublas_v2.h>
#include <cuda.h>

#include <iostream>
#include <string>

#include "cuda_util.h"

class Context {
 public:
  Context() : _stream(nullptr) {
    CHECK_GPU_ERROR(cublasCreate(&_cublasHandle));
  }

  virtual ~Context() {}

  static Context &Instance() {
    static Context _ctx;
    return _ctx;
  }

  void set_stream(cudaStream_t stream) {
    _stream = stream;
    CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
  }

  cudaStream_t get_stream() { return _stream; }

  cublasHandle_t get_cublashandle() { return _cublasHandle; }

 private:
  cudaStream_t _stream;
  cublasHandle_t _cublasHandle;
};
