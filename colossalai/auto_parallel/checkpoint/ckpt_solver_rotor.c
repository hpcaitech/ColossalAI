#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
Rotor solver for checkpointing problem in C. We follow the modeling mentioned in
paper `Optimal checkpointing for heterogeneous chains: how to train deep neural
networks with limited memory` https://hal.inria.fr/hal-02352969. Some lines of
the code are adapted from https://gitlab.inria.fr/hiepacs/rotor.
*/
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

static PyObject* computeTable(PyObject* self, PyObject* args) {
  PyObject* chainParam;
  int mmax;

  if (!PyArg_ParseTuple(args, "Oi", &chainParam, &mmax)) return NULL;

  double* ftime = getDoubleArray(chainParam, "ftime");
  if (!ftime) return NULL;

  double* btime = getDoubleArray(chainParam, "btime");
  if (!btime) return NULL;

  long* x = getLongArray(chainParam, "x");
  if (!x) return NULL;

  long* xbar = getLongArray(chainParam, "xbar");
  if (!xbar) return NULL;

  long* ftmp = getLongArray(chainParam, "btmp");
  if (!ftmp) return NULL;

  long* btmp = getLongArray(chainParam, "btmp");
  if (!btmp) return NULL;

  long chainLength = PyObject_Length(chainParam);
  if (!chainLength) return NULL;

#define COST_TABLE(m, i, l)                               \
  costTable[(m) * (chainLength + 1) * (chainLength + 1) + \
            (i) * (chainLength + 1) + (l)]
  double* costTable = (double*)calloc(
      (mmax + 1) * (chainLength + 1) * (chainLength + 1), sizeof(double));

#define BACK_PTR(m, i, l)                               \
  backPtr[(m) * (chainLength + 1) * (chainLength + 1) + \
          (i) * (chainLength + 1) + (l)]
  long* backPtr = (long*)calloc(
      (mmax + 1) * (chainLength + 1) * (chainLength + 1), sizeof(long));

  for (long m = 0; m <= mmax; ++m)
    for (long i = 0; i <= chainLength; ++i) {
      if ((m >= x[i + 1] + xbar[i + 1] + btmp[i]) &&
          (m >= x[i + 1] + xbar[i + 1] + ftmp[i])) {
        COST_TABLE(m, i, i) = ftime[i] + btime[i];
      } else {
        COST_TABLE(m, i, i) = INFINITY;
      }
    }

  for (long m = 0; m <= mmax; ++m) {
    for (long d = 1; d <= chainLength; ++d) {
      for (long i = 0; i <= chainLength - d; ++i) {
        long idx = i + d;
        long mmin = x[idx + 1] + x[i + 1] + ftmp[i];
        if (idx > i + 1) {
          long maxCostFWD = 0;
          for (long j = i + 1; j < idx; j++) {
            maxCostFWD = fmaxl(maxCostFWD, x[j] + x[j + 1] + ftmp[j]);
          }
          mmin = fmaxl(mmin, x[idx + 1] + maxCostFWD);
        }
        if ((m >= mmin)) {
          long bestLeaf = -1;
          double sumFw = 0;
          double bestLeafCost = INFINITY;
          for (long j = i + 1; j <= idx; ++j) {
            sumFw += ftime[j - 1];
            if (m >= x[j]) {
              double cost = sumFw + COST_TABLE(m - x[j], j, idx) +
                            COST_TABLE(m, i, j - 1);
              if (cost < bestLeafCost) {
                bestLeafCost = cost;
                bestLeaf = j;
              }
            }
          }
          double chainCost = INFINITY;
          if (m >= xbar[i + 1]) {
            chainCost =
                COST_TABLE(m, i, i) + COST_TABLE(m - xbar[i + 1], i + 1, idx);
          }
          if (bestLeafCost <= chainCost) {
            COST_TABLE(m, i, idx) = bestLeafCost;
            BACK_PTR(m, i, idx) = bestLeaf;
          } else {
            COST_TABLE(m, i, idx) = chainCost;
            BACK_PTR(m, i, idx) = -1;
          }
        } else {
          COST_TABLE(m, i, idx) = INFINITY;
        }
      }
    }
  }

  free(ftime);
  free(btime);
  free(x);
  free(xbar);
  free(ftmp);
  free(btmp);

  PyObject* pyCostTable = PyList_New(mmax + 1);
  PyObject* pyBackPtr = PyList_New(mmax + 1);

  // Convert the result into Python world
  for (long m = 0; m <= mmax; ++m) {
    PyObject* pyCostTable_m = PyList_New(chainLength + 1);
    PyList_SET_ITEM(pyCostTable, m, pyCostTable_m);
    PyObject* pyBackPtr_m = PyList_New(chainLength + 1);
    PyList_SET_ITEM(pyBackPtr, m, pyBackPtr_m);
    for (long i = 0; i <= chainLength; ++i) {
      PyObject* pyCostTable_m_i = PyDict_New();
      PyList_SET_ITEM(pyCostTable_m, i, pyCostTable_m_i);
      PyObject* pyBackPtr_m_i = PyDict_New();
      PyList_SET_ITEM(pyBackPtr_m, i, pyBackPtr_m_i);
      for (long l = i; l <= chainLength; ++l) {
        PyObject* pyVar_l = PyLong_FromLong(l);
        PyObject* pyCostTable_m_i_l = PyFloat_FromDouble(COST_TABLE(m, i, l));
        PyDict_SetItem(pyCostTable_m_i, pyVar_l, pyCostTable_m_i_l);
        Py_DECREF(pyCostTable_m_i_l);
        PyObject* pyBackPtr_m_i_l;
        if (BACK_PTR(m, i, l) < 0) {
          pyBackPtr_m_i_l = Py_BuildValue("(O)", Py_True);
        } else {
          pyBackPtr_m_i_l = Py_BuildValue("(Ol)", Py_False, BACK_PTR(m, i, l));
        }
        PyDict_SetItem(pyBackPtr_m_i, pyVar_l, pyBackPtr_m_i_l);
        Py_DECREF(pyBackPtr_m_i_l);
        Py_DECREF(pyVar_l);
      }
    }
  }

  free(costTable);
  free(backPtr);

  PyObject* result = PyTuple_Pack(2, pyCostTable, pyBackPtr);
  Py_DECREF(pyCostTable);
  Py_DECREF(pyBackPtr);
  return result;
}

static PyMethodDef rotorMethods[] = {
    {"compute_table", computeTable, METH_VARARGS,
     "Compute the optimal table with the rotor algorithm."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef rotorModule = {
    PyModuleDef_HEAD_INIT, "rotorc", /* name of module */
    "A simple implementation of dynamic programming algorithm rotor with C in "
    "https://hal.inria.fr/hal-02352969. Some code are adapted from "
    "https://gitlab.inria.fr/hiepacs/rotor.", /* module documentation, may be
                                                 NULL */
    -1, /* size of per-interpreter state of the module,
                   or -1 if the module keeps state in global variables. */
    rotorMethods};

PyMODINIT_FUNC PyInit_rotorc(void) { return PyModule_Create(&rotorModule); }
