# cython methods to speed-up evaluation

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()

cdef extern from "addToConfusionMatrix_impl.c":
	void addToConfusionMatrix( const unsigned char* f_prediction_p  ,
                               const unsigned char* f_groundTruth_p ,
                               const unsigned int   f_width_i       ,
                               const unsigned int   f_height_i      ,
                               unsigned long long*  f_confMatrix_p  ,
                               const unsigned int   f_confMatDim_i  )


cdef tonumpyarray(unsigned long long* data, unsigned long long size):
	if not (data and size >= 0): raise ValueError
	return np.PyArray_SimpleNewFromData(2, [size, size], np.NPY_UINT64, <void*>data)

@cython.boundscheck(False)
def cEvaluatePair( np.ndarray[np.uint8_t , ndim=2] predictionArr   ,
                   np.ndarray[np.uint8_t , ndim=2] groundTruthArr  ,
                   np.ndarray[np.uint64_t, ndim=2] confMatrix      ,
                   evalLabels                                    ):
	cdef np.ndarray[np.uint8_t    , ndim=2, mode="c"] predictionArr_c
	cdef np.ndarray[np.uint8_t    , ndim=2, mode="c"] groundTruthArr_c
	cdef np.ndarray[np.ulonglong_t, ndim=2, mode="c"] confMatrix_c

	predictionArr_c  = np.ascontiguousarray(predictionArr , dtype=np.uint8    )
	groundTruthArr_c = np.ascontiguousarray(groundTruthArr, dtype=np.uint8    )
	confMatrix_c     = np.ascontiguousarray(confMatrix    , dtype=np.ulonglong)

	cdef np.uint32_t height_ui     = predictionArr.shape[1]
	cdef np.uint32_t width_ui      = predictionArr.shape[0]
	cdef np.uint32_t confMatDim_ui = confMatrix.shape[0]

	addToConfusionMatrix(&predictionArr_c[0,0], &groundTruthArr_c[0,0], height_ui, width_ui, &confMatrix_c[0,0], confMatDim_ui)

	confMatrix = np.ascontiguousarray(tonumpyarray(&confMatrix_c[0,0], confMatDim_ui))

	return np.copy(confMatrix)