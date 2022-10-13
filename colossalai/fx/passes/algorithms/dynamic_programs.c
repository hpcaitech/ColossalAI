#define PY_SSIZE_T_CLEAN
#include <Python.h>

long* PySequenceToLongArray(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist))) return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  long* result = (long*)calloc(len + 1, sizeof(long));
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PyLong_AsLong(item);
    Py_DECREF(item);
  }
  result[len] = 0;
  return result;
}

double* PySequenceToDoubleArray(PyObject* pylist) {
  if (!(pylist && PySequence_Check(pylist))) return NULL;
  Py_ssize_t len = PySequence_Size(pylist);
  double* result = (double*)calloc(len + 1, sizeof(double));
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_GetItem(pylist, i);
    result[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
  }
  result[len] = 0;
  return result;
}

long* getLongArray(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName);
  long* result = PySequenceToLongArray(sequence);
  Py_DECREF(sequence);
  return result;
}

double* getDoubleArray(PyObject* container, const char* attributeName) {
  PyObject* sequence = PyObject_GetAttrString(container, attributeName);
  double* result = PySequenceToDoubleArray(sequence);
  Py_DECREF(sequence);
  return result;
}

static PyObject* persistent_compute_table(PyObject* self, PyObject* args) {
  PyObject* chain_param;
  int mmax;

  if (!PyArg_ParseTuple(args, "Oi", &chain_param, &mmax)) return NULL;

  double* fw = getDoubleArray(chain_param, "fweight");
  if (!fw) return NULL;

  double* bw = getDoubleArray(chain_param, "bweight");
  if (!bw) return NULL;

  long* cw = getLongArray(chain_param, "cweight");
  if (!cw) return NULL;

  long* cbw = getLongArray(chain_param, "cbweight");
  if (!cbw) return NULL;

  long* fwd_tmp = getLongArray(chain_param, "fwd_mem_tmp");
  if (!cbw) return NULL;

  long* bwd_tmp = getLongArray(chain_param, "bwd_mem_tmp");
  if (!cbw) return NULL;

  PyObject* chain_length_param = PyObject_GetAttrString(chain_param, "length");
  if (!chain_length_param) return NULL;
  long chain_length = PyLong_AsLong(chain_length_param);
  Py_DECREF(chain_length_param);

  // TODO: Can be optimized by only allocating memory for l >= i
  // TODO: float / int instead of double / long ?
#define OPT(m, i, l)                                  \
  opt[(m) * (chain_length + 1) * (chain_length + 1) + \
      (i) * (chain_length + 1) + (l)]
  double* opt = (double*)calloc(
      (mmax + 1) * (chain_length + 1) * (chain_length + 1), sizeof(double));

#define WHAT(m, i, l)                                  \
  what[(m) * (chain_length + 1) * (chain_length + 1) + \
       (i) * (chain_length + 1) + (l)]
  long* what = (long*)calloc(
      (mmax + 1) * (chain_length + 1) * (chain_length + 1), sizeof(long));

  for (long m = 0; m <= mmax; ++m)
    for (long i = 0; i <= chain_length; ++i)
      // TODO: Can be optimized to remove the IF by reordering loops
      if ((m >= cw[i + 1] + cbw[i + 1] + bwd_tmp[i]) &&
          (m >= cw[i + 1] + cbw[i + 1] + fwd_tmp[i]))
        OPT(m, i, i) = fw[i] + bw[i];
      else
        OPT(m, i, i) = INFINITY;

  for (long m = 0; m <= mmax; ++m)
    for (long d = 1; d <= chain_length; ++d) {
      for (long i = 0; i <= chain_length - d; ++i) {
        long idx = i + d;
        long mmin = cw[idx + 1] + cw[i + 1] + fwd_tmp[i];
        if (idx > i + 1) {
          long maxCostFWD = 0;
          for (long j = i + 1; j < idx; j++) {
            maxCostFWD = fmaxl(maxCostFWD, cw[j] + cw[j + 1] + fwd_tmp[j]);
          }
          mmin = fmaxl(mmin, cw[idx + 1] + maxCostFWD);
        }
        if ((m >= mmin)) {
          long bestLeaf = -1;
          double sumFw = 0;
          double bestLeafCost = INFINITY;
          /// sumFw + OPT(m-cw[i+1], i+1, l) + OPT(m, i, i); // Value for j =
          /// i+1
          for (long j = i + 1; j <= idx; ++j) {
            sumFw += fw[j - 1];
            if (m >= cw[j]) {
              double cost = sumFw + OPT(m - cw[j], j, idx) + OPT(m, i, j - 1);
              if (cost < bestLeafCost) {
                bestLeafCost = cost;
                bestLeaf = j;
              }
            }
          }
          double chainCost = INFINITY;
          if (m >= cbw[i + 1])
            chainCost = OPT(m, i, i) + OPT(m - cbw[i + 1], i + 1, idx);
          if (bestLeafCost <= chainCost) {
            OPT(m, i, idx) = bestLeafCost;
            WHAT(m, i, idx) = bestLeaf;
          } else {
            OPT(m, i, idx) = chainCost;
            WHAT(m, i, idx) = -1;
          }
        } else
          OPT(m, i, idx) = INFINITY;
      }
    }

  free(fw);
  free(bw);
  free(cw);
  free(cbw);
  free(fwd_tmp);
  free(bwd_tmp);

  PyObject* res_opt = PyList_New(mmax + 1);
  PyObject* res_what = PyList_New(mmax + 1);

  // Convert the result into Python world
  for (long m = 0; m <= mmax; ++m) {
    PyObject* res_opt_m = PyList_New(chain_length + 1);
    PyList_SET_ITEM(res_opt, m, res_opt_m);
    PyObject* res_what_m = PyList_New(chain_length + 1);
    PyList_SET_ITEM(res_what, m, res_what_m);
    for (long i = 0; i <= chain_length; ++i) {
      PyObject* res_opt_m_i = PyDict_New();
      PyList_SET_ITEM(res_opt_m, i, res_opt_m_i);
      PyObject* res_what_m_i = PyDict_New();
      PyList_SET_ITEM(res_what_m, i, res_what_m_i);
      for (long l = i; l <= chain_length; ++l) {
        PyObject* res_l = PyLong_FromLong(l);
        PyObject* res_opt_m_i_l = PyFloat_FromDouble(OPT(m, i, l));
        PyDict_SetItem(res_opt_m_i, res_l, res_opt_m_i_l);
        Py_DECREF(res_opt_m_i_l);
        PyObject* res_what_m_i_l;
        long what_m_i_l = WHAT(m, i, l);
        if (what_m_i_l < 0)
          res_what_m_i_l = Py_BuildValue("(O)", Py_True);
        else
          res_what_m_i_l = Py_BuildValue("(Ol)", Py_False, what_m_i_l);
        PyDict_SetItem(res_what_m_i, res_l, res_what_m_i_l);
        Py_DECREF(res_what_m_i_l);
        Py_DECREF(res_l);
      }
    }
  }

  free(opt);
  free(what);

  PyObject* result = PyTuple_Pack(2, res_opt, res_what);
  Py_DECREF(res_opt);
  Py_DECREF(res_what);
  return result;
}

//  long i = L - s, j = t - s, k = l - t
inline long floating_index_in_array(long m_factor, long m, long i, long j,
                                    long k) {
  return m * m_factor + (i * (i + 1) * (2 * i + 4)) / 12 + (i + 1) * j -
         (j * (j - 1)) / 2 + k;
}

typedef struct {
  long sp;
  long r;
  long tp;
} index_t;

static PyObject* floating_compute_table(PyObject* self, PyObject* args) {
  PyObject* chain_param;
  int mmax;

  if (!PyArg_ParseTuple(args, "Oi", &chain_param, &mmax)) return NULL;

  double* fw = getDoubleArray(chain_param, "fweigth");
  if (!fw) return NULL;

  double* bw = getDoubleArray(chain_param, "bweigth");
  if (!bw) return NULL;

  long* cw = getLongArray(chain_param, "cweigth");
  if (!cw) return NULL;

  long* cbw = getLongArray(chain_param, "cbweigth");
  if (!cbw) return NULL;

  long* fwd_tmp = getLongArray(chain_param, "fwd_tmp");
  if (!fwd_tmp) return NULL;

  long* bwd_tmp = getLongArray(chain_param, "bwd_tmp");
  if (!bwd_tmp) return NULL;

  PyObject* chain_length_param = PyObject_GetAttrString(chain_param, "length");
  if (!chain_length_param) return NULL;
  long chain_length = PyLong_AsLong(chain_length_param);
  Py_DECREF(chain_length_param);

  const long m_factor =
      (chain_length + 1) * (chain_length + 2) * (2 * chain_length + 6) / 12;

  // Defined for 0 <= s <= t <= l <= chain_length, for all m
#undef OPT
#define OPT(m, s, t, l)                                                     \
  opt[floating_index_in_array(m_factor, (m), chain_length - (s), (t) - (s), \
                              (l) - (t))]
  double* opt = (double*)calloc((mmax + 1) * m_factor, sizeof(double));

#undef WHAT
#define WHAT(m, s, t, l)                                                     \
  what[floating_index_in_array(m_factor, (m), chain_length - (s), (t) - (s), \
                               (l) - (t))]
  index_t* what = (index_t*)calloc((mmax + 1) * m_factor, sizeof(index_t));

  double* partialSumsFW = (double*)calloc(chain_length + 1, sizeof(double));
  double total = 0;
  for (long i = 0; i < chain_length; ++i) {
    partialSumsFW[i] = total;
    total += fw[i];
  }
  partialSumsFW[chain_length] = total;

  for (long m = 0; m <= mmax; ++m)
    for (long i = 0; i <= chain_length; ++i) {
      // TODO: Can be optimized to remove the IF by reordering loops
      if ((m >= cw[i] + cw[i + 1] + cbw[i + 1] + bwd_tmp[i]) &&
          (m >= cw[i + 1] + cbw[i + 1] + fwd_tmp[i]))
        OPT(m, i, i, i) = fw[i] + bw[i];
      else
        OPT(m, i, i, i) = INFINITY;
    }

  for (long m = 0; m <= mmax; ++m)
    for (long d = 1; d <= chain_length; ++d) {  // d = l - s
      for (long s = 0; s <= chain_length - d; ++s) {
        long l = s + d;
        long memNullFirst = cw[l + 1] + cw[s + 1] + fwd_tmp[s];
        long memNullSecond = 0;
        for (long j = s + 1; j < l; ++j) {
          long val = cw[j] + cw[j + 1] + fwd_tmp[j];
          if (val > memNullSecond) memNullSecond = val;
        }
        for (long t = s; t <= l; ++t) {
          double chainCost = INFINITY;
          if ((s == t) && (m >= cw[l + 1] + cbw[s + 1] + fwd_tmp[s]) &&
              (m >= cw[s] + cw[s + 1] + cbw[s + 1] + bwd_tmp[s])) {
            chainCost = OPT(m, s, s, s) + OPT(m - cbw[s + 1], s + 1, s + 1, l);
          }
          double bestLeafCost = INFINITY;
          index_t bestLeaf = {.sp = -1, .r = -1, .tp = -1};
          if (m >= memNullFirst && m >= cw[l + 1] + memNullSecond) {
            for (long r = s; r <= t; ++r)
              if (cw[s] <= cw[r])
                for (long tp = t + 1; tp <= l; ++tp)
                  for (long sp = r + 1; sp <= tp; ++sp) {
                    long mp = m - cw[r] + cw[s];
                    assert(mp >= 0);
                    if (mp >= cw[sp]) {
                      double value = partialSumsFW[sp] - partialSumsFW[s] +
                                     OPT(mp - cw[sp], sp, tp, l) +
                                     OPT(mp, r, t, tp - 1);
                      if (value < bestLeafCost) {
                        bestLeafCost = value;
                        bestLeaf.sp = sp;
                        bestLeaf.r = r;
                        bestLeaf.tp = tp;
                      }
                    }
                  }
          }
          if (bestLeaf.sp >= 0 && bestLeafCost <= chainCost) {
            OPT(m, s, t, l) = bestLeafCost;
            WHAT(m, s, t, l).sp = bestLeaf.sp;
            WHAT(m, s, t, l).r = bestLeaf.r;
            WHAT(m, s, t, l).tp = bestLeaf.tp;
          } else {
            OPT(m, s, t, l) = chainCost;
            WHAT(m, s, t, l).sp = -1;
          }
        }
      }
    }

  free(fw);
  free(bw);
  free(cw);
  free(cbw);
  free(fwd_tmp);
  free(bwd_tmp);

  PyObject* res_opt = PyList_New(mmax + 1);
  PyObject* res_what = PyList_New(mmax + 1);

  // Convert the result into Python world
  PyObject* true_tuple = Py_BuildValue("(O)", Py_True);
  for (long m = 0; m <= mmax; ++m) {
    PyObject* res_opt_m = PyDict_New();
    PyList_SET_ITEM(res_opt, m, res_opt_m);
    PyObject* res_what_m = PyDict_New();
    PyList_SET_ITEM(res_what, m, res_what_m);
    for (long s = 0; s <= chain_length; ++s)
      for (long t = s; t <= chain_length; ++t)
        for (long l = t; l <= chain_length; ++l) {
          PyObject* key = Py_BuildValue("(lll)", s, t, l);
          PyObject* value_opt = PyFloat_FromDouble(OPT(m, s, t, l));
          PyDict_SetItem(res_opt_m, key, value_opt);
          PyObject* value_what = true_tuple;
          index_t* idx_what = &WHAT(m, s, t, l);
          if (idx_what->sp >= 0)
            value_what = Py_BuildValue("(O(lll))", Py_False, idx_what->sp,
                                       idx_what->r, idx_what->tp);
          PyDict_SetItem(res_what_m, key, value_what);
          if (value_what != true_tuple) Py_DECREF(value_what);
          Py_DECREF(key);
          Py_DECREF(value_opt);
        }
  }

  Py_DECREF(true_tuple);

  free(opt);
  free(what);

  PyObject* result = PyTuple_Pack(2, res_opt, res_what);
  Py_DECREF(res_opt);
  Py_DECREF(res_what);
  return result;
}

static PyObject* griewank_heterogeneous_compute_table(PyObject* self,
                                                      PyObject* args) {
  PyObject* chain_param;
  int mmax;

  if (!PyArg_ParseTuple(args, "Oi", &chain_param, &mmax)) return NULL;

  double* fw = getDoubleArray(chain_param, "fweigth");
  if (!fw) return NULL;

  double* bw = getDoubleArray(chain_param, "bweigth");
  if (!bw) return NULL;

  long* cw = getLongArray(chain_param, "cweigth");
  if (!cw) return NULL;

  long* cbw = getLongArray(chain_param, "cbweigth");
  if (!cbw) return NULL;

  PyObject* chain_length_param = PyObject_GetAttrString(chain_param, "length");
  if (!chain_length_param) return NULL;
  long chain_length = PyLong_AsLong(chain_length_param);
  Py_DECREF(chain_length_param);

  // TODO: Can be optimized by only allocating memory for l >= i
  // TODO: float / int instead of double / long ?
#undef OPT
#define OPT(m, i, l)                                  \
  opt[(m) * (chain_length + 1) * (chain_length + 1) + \
      (i) * (chain_length + 1) + (l)]
  double* opt = (double*)calloc(
      (mmax + 1) * (chain_length + 1) * (chain_length + 1), sizeof(double));

  // Compute partial sums
  double* sumfw = (double*)calloc(chain_length, sizeof(double));
  double* sumbw = (double*)calloc(chain_length + 1, sizeof(double));
  double* sumsumfw = (double*)calloc(chain_length, sizeof(double));

  double total = 0;
  for (long i = 0; i < chain_length; ++i) {
    total += fw[i];
    sumfw[i] = total;
  }

  total = 0;
  for (long i = 0; i < chain_length + 1; ++i) {
    total += bw[i];
    sumbw[i] = total;
  }

  total = 0;
  for (long i = 0; i < chain_length; ++i) {
    total += sumfw[i];
    sumsumfw[i] = total;
  }

  for (long m = 0; m <= mmax; ++m)
    for (long i = 0; i <= chain_length; ++i) {
      // TODO: Can be optimized to remove the IF by reordering loops
      if ((m >= cbw[i]) && (m >= cw[i] + cbw[i + 1]))
        OPT(m, i, i) = bw[i];
      else
        OPT(m, i, i) = INFINITY;

      if (i < chain_length) {
        long maxC = fmaxl(cw[i], cw[i + 1]);
        long maxCB = fmaxl(cbw[i + 1], cbw[i + 2] + maxC);
        if ((m >= cbw[i]) && (m >= cw[i] + maxCB))
          OPT(m, i, i + 1) = fw[i] + bw[i] + bw[i + 1];
        else
          OPT(m, i, i + 1) = INFINITY;
      }
    }

  for (long m = 0; m <= mmax; ++m)
    for (long i = 0; i + 2 <= chain_length; ++i) {
      long mminCst = fmaxl(cbw[i], cbw[i + 1] + cw[i]);
      long maxCW_il = fmax(fmax(cw[i], cw[i + 1]), cw[i + 2]);
      long maxCostFWD = cw[i] + cbw[i + 2] + maxCW_il;
      for (long l = i + 2; l <= chain_length; ++l) {
        maxCW_il = fmax(maxCW_il, cw[l + 1]);
        maxCostFWD = fmaxl(maxCostFWD, cw[i] + cw[l + 1] + maxCW_il);
        long mmin = fmaxl(mminCst, maxCostFWD);
        if ((m >= mmin)) {
          double noCheckpointCost = sumbw[l] - (i > 0 ? sumbw[i - 1] : 0);
          noCheckpointCost +=
              sumsumfw[l - 1] -
              (i > 0 ? sumsumfw[i - 1] + (l - i) * sumfw[i - 1] : 0);

          double valueCost = INFINITY;
          if (m >= cw[i]) {
            double sumFwds = 0;
            for (long j = i + 1; j < l; ++j) {
              sumFwds += fw[j - 1];
              valueCost = fmin(
                  valueCost, sumFwds + OPT(m - cw[i], j, l) + OPT(m, i, j - 1));
            }
          }
          OPT(m, i, l) = fmin(noCheckpointCost, valueCost);
        } else
          OPT(m, i, l) = INFINITY;
      }
    }

  free(sumfw);
  free(sumbw);
  free(sumsumfw);
  free(fw);
  free(bw);
  free(cw);
  free(cbw);

  PyObject* res_opt = PyList_New(mmax + 1);

  // Convert the result into Python world
  for (long m = 0; m <= mmax; ++m) {
    PyObject* res_opt_m = PyList_New(chain_length + 1);
    PyList_SET_ITEM(res_opt, m, res_opt_m);
    for (long i = 0; i <= chain_length; ++i) {
      PyObject* res_opt_m_i = PyDict_New();
      PyList_SET_ITEM(res_opt_m, i, res_opt_m_i);
      for (long l = i; l <= chain_length; ++l) {
        PyObject* res_l = PyLong_FromLong(l - i);
        PyObject* res_opt_m_i_l = PyFloat_FromDouble(OPT(m, i, l));
        PyDict_SetItem(res_opt_m_i, res_l, res_opt_m_i_l);
        Py_DECREF(res_opt_m_i_l);
        Py_DECREF(res_l);
      }
    }
  }

  free(opt);

  return res_opt;
}

static PyMethodDef dynamic_programs_methods[] = {
    {"persistent_compute_table", persistent_compute_table, METH_VARARGS,
     "Compute the optimal table with the persistent algorithm."},
    {"floating_compute_table", floating_compute_table, METH_VARARGS,
     "Compute the optimal table with the floating algorithm."},
    {"griewank_heterogeneous_compute_table",
     griewank_heterogeneous_compute_table, METH_VARARGS,
     "Compute the optimal table for the Griewank Heterogeneous Model."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef dynamic_programs_module = {
    PyModuleDef_HEAD_INIT, "dynamic_programs_C_version", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
    dynamic_programs_methods};

PyMODINIT_FUNC PyInit_dynamic_programs_C_version(void) {
  return PyModule_Create(&dynamic_programs_module);
}
