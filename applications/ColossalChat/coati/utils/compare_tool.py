########################
# Reference from https://github.com/MooreThreads/torch_musa/blob/main/torch_musa/utils/compare_tool.py
# A verion for gpu and npu
# BSD 3-Clause License
# Copyright (c) 2023 , Moore Threads Technology  Co., Ltd.
# Copyright (c) 2022, Facebook Inc. and the respective contributors
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------------
# This product bundles various third-party components under other open source licenses.
# This section summarizes those components and their licenses. See licenses/
# for text of these licenses.

# License in PyToch(https://github.com/pytorch/pytorch/blob/main/LICENSE)
# -----------------
# tools/setup_helper
# torch_musa/csrc/
# torch_musa/core
# Apache Software Foundation License 2.0
# --------------------------------------
# tools/lint
# BSD 2-clause License
# --------------------
# docs
# Apache Software Foundation License 2.0
# --------------------------------------
# examples/cpp
########################
"compare tool with cpu"
# pylint: disable=broad-exception-caught,broad-exception-raised,redefined-builtin,unused-argument
import os
import pickle
import sys
from datetime import datetime
from functools import partial

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily


class ModuleInfo(object):
    """
    A class to store information about a module in a neural network, including its name,
    relationship to other modules (parent and children), and whether the module is a leaf
    or is being executed in a forward or backward pass.
    """

    def __init__(self, name, father, is_leaf=False, is_forward=True) -> None:
        self.name = name  # Name of the module
        self.father = father  # Parent module in the hierarchy
        self.children = []  # List of child modules
        self.is_leaf = is_leaf  # Flag indicating if the module is a leaf module
        self.is_forward = is_forward  # Flag indicating if the module is in the forward pass

    def name_with_prefix(self):
        """
        Constructs the module's full name with hierarchical prefix based on parent names.

        Returns:
        str: The full hierarchical name of the module.
        """
        if self.father is None or self.father.name_with_prefix() == "":
            prefix = ""
        else:
            prefix = self.father.name_with_prefix() + "/"
        return prefix + self.name

    def full_name(self):
        """
        Generates the module's full name with an additional suffix indicating
        whether the current state is forward or backward pass.

        Returns:
        str: The full name of the module including its forward/backward state.
        """
        is_forward = self.is_forward if not self.is_leaf else self.father.is_forward
        suffix = "(forward)" if is_forward else "(backward)"
        return self.name_with_prefix() + suffix


# Initialize root module info as the base of the module hierarchy
root_module_info = ModuleInfo(name="", father=None)
current_module_info = root_module_info


def pre_forward_hook(module, input):
    """
    Hook to be executed before a module's forward pass. It updates the module hierarchy
    by adding the current module as a child of the current module in the hierarchy.

    Parameters:
    - module: The module where the hook is registered.
    - input: The input to the forward method of the module.
    """
    global current_module_info
    module_info = ModuleInfo(module.__class__.__name__, father=current_module_info, is_forward=True)
    current_module_info.children.append(module_info)
    current_module_info = module_info


def post_forward_hook(module, input, output):
    """
    Hook to be executed after a module's forward pass. It steps back in the module hierarchy
    to the parent module.

    Parameters:
    - module: The module where the hook is registered.
    - input: The input to the forward method of the module.
    - output: The output from the forward method of the module.
    """
    global current_module_info
    current_module_info = current_module_info.father


def pre_backward_hook(module, grad_output):
    """
    Hook to be executed before a module's backward pass. Similar to the pre_forward_hook,
    it adds the module to the hierarchy with an indication that it's part of the backward pass.

    Parameters:
    - module: The module where the hook is registered.
    - grad_output: The gradients at the output of the module.
    """
    global current_module_info
    module_info = ModuleInfo(module.__class__.__name__, current_module_info, is_forward=False)
    current_module_info.children.append(module_info)
    current_module_info = module_info


def post_backward_hook(module, grad_input, grad_output):
    """
    Hook to be executed after a module's backward pass.
    It steps back to the parent module in the hierarchy.

    Parameters:
    - module: The module where the hook is registered.
    - grad_input: The gradients at the input of the module.
    - grad_output: The gradients at the output of the module.
    """
    global current_module_info
    current_module_info = current_module_info.father


def register_hooks(module):
    """
    Registers the forward and backward hooks on the module and all its submodules.

    Parameters:
    - module: The root module to register hooks on.
    """
    module.register_forward_pre_hook(pre_forward_hook)
    module.register_forward_hook(post_forward_hook)
    module.register_full_backward_pre_hook(pre_backward_hook)
    module.register_full_backward_hook(post_backward_hook)


def open_module_tracker(module):
    """
    Initializes the module tracking by applying the register_hooks function to the module
    and all its submodules.

    Parameters:
    - module: The root module to start tracking on.
    """
    module.apply(register_hooks)


def recursive_apply(func):
    """
    Applies a function recursively to all tensors in a nested structure of
    tensors, lists, tuples, and dictionaries.

    Parameters:
    - func (function): A function to apply to every tensor found in the input structure.

    Returns:
    - A function that takes an input structure and applies 'func'
        to every tensor within that structure.
    """

    def recursive_apply_fn(inputs):
        if isinstance(inputs, (list, tuple)):
            # Recursively apply to each element in lists or tuples
            inputs_dst = [None] * len(inputs)
            for i, x in enumerate(inputs):
                inputs_dst[i] = recursive_apply_fn(x)
            return tuple(inputs_dst) if isinstance(inputs, tuple) else inputs_dst
        if isinstance(inputs, dict):
            # Recursively apply to each value in dictionaries
            return {k: recursive_apply_fn(v) for k, v in inputs.items()}
        if isinstance(inputs, torch.Tensor):
            # Apply the function to tensors
            return func(inputs.detach().clone())
        # Return non-tensor objects unchanged
        return inputs

    return recursive_apply_fn


def convert_to_dtype(inputs, dtype):
    """
    Converts all tensors in a nested structure to a specified data type.

    Parameters:
    - inputs: The input structure containing tensors.
    - dtype: The target data type.

    Returns:
    - The input structure with all tensors converted to the specified data type.
    """
    return recursive_apply(lambda x: x.to(dtype=dtype))(inputs)


def convert_to_cpu(inputs):
    """
    Converts all tensors in a nested structure to CPU memory.

    Parameters:
    - inputs: The input structure containing tensors.

    Returns:
    - The input structure with all tensors moved to CPU memory.
    """
    return recursive_apply(lambda x: x.cpu())(inputs)


def convert_to_npu(inputs):
    """
    Converts all tensors in a nested structure to a specified device,
     in this case, a fictional "npu" device.

    Parameters:
    - inputs: The input structure containing tensors.

    Returns:
    - The input structure with all tensors moved to the "npu" device.
    """
    return recursive_apply(lambda x: x.to("npu"))(inputs)


def convert_to_gpu(inputs):
    """
    Converts all tensors in a nested structure to a specified device,
     in this case, a fictional "npu" device.

    Parameters:
    - inputs: The input structure containing tensors.

    Returns:
    - The input structure with all tensors moved to the "npu" device.
    """
    return recursive_apply(lambda x: x.to("cuda"))(inputs)


def compare_tensors(tensor1, tensor2, atol, rtol):
    """
    Compares two tensors element-wise to check if they are approximately
    equal within the given tolerances,
    and returns a mask indicating where they differ.

    Parameters:
    - tensor1, tensor2: The tensors to compare.
    - atol (float): Absolute tolerance.
    - rtol (float): Relative tolerance.

    Returns:
    - A boolean tensor mask where True indicates positions
    where tensor1 and tensor2 are not close.
    """
    # Initial comparisons for NaN and Inf values
    nan_mask1, nan_mask2 = torch.isnan(tensor1), torch.isnan(tensor2)
    inf_mask1, inf_mask2 = torch.isinf(tensor1), torch.isinf(tensor2)

    # Identify differences in NaN and Inf
    nan_diff = nan_mask1 != nan_mask2
    inf_diff = inf_mask1 != inf_mask2

    # Calculate the difference and tolerance
    basic_diff = torch.abs(tensor1 - tensor2)
    tolerance = atol + rtol * torch.abs(tensor2)

    # Check normal (non-NaN, non-Inf) value differences
    normal_not_close = (basic_diff > tolerance) & ~(nan_mask1 | inf_mask1)

    # Aggregate differences
    not_close = nan_diff | inf_diff | normal_not_close

    return not_close


def format_tensor(tensor):
    """
    Formats a tensor for printing, providing information about
    NaN and Inf values, shape, and other properties.

    Parameters:
    - tensor: The tensor to format.

    Returns:
    - A string representing the formatted tensor information.
    """
    nan_num = torch.isnan(tensor).sum().item()
    inf_num = torch.isinf(tensor).sum().item()
    head = "[WARNING] " if nan_num or inf_num else ""
    warnings = f"nan_num={nan_num}, inf_num={inf_num}" if nan_num or inf_num else ""
    return (
        f"{head}Tensor <shape={tensor.shape}, dtype={tensor.dtype}, "
        f"stride={tensor.stride()}, is_contiguous={tensor.is_contiguous()}, "
        f"device={tensor.device}, size={tensor.numel()}, {warnings}>"
    )


def print_tensors_diff(tensor1, tensor2, atol, rtol):
    """
    Prints the differences between two tensors, indicating
    where they do not match within the given tolerances.

    Parameters:
    - tensor1, tensor2: The tensors to compare.
    - atol (float): Absolute tolerance.
    - rtol (float): Relative tolerance.

    Returns:
    - A string detailing the indices and values where the tensors differ.
    """
    not_close = compare_tensors(tensor1.to(tensor2.device).to(tensor2.dtype), tensor2, atol, rtol)
    indices = torch.nonzero(not_close)
    indices_np = indices.cpu().numpy()
    diff_str = ""
    # If the indices are too large, only process the front part
    if len(indices_np) > 20:
        diff_str += f"\nToo many indices (total {len(indices_np)}) to print \n\n...\n\n"
        indices_np = indices_np[:20]
    idx_tuples = [tuple(idx) for idx in indices_np]
    elements_out1 = [tensor1[idx].item() for idx in idx_tuples]
    elements_out2 = [tensor2[idx].item() for idx in idx_tuples]

    for idx_tuple, elem1, elem2 in zip(idx_tuples, elements_out1, elements_out2):
        diff_str += (
            f"Element at index {idx_tuple} is not close:"
            f"{elem1}({tensor1.device}) vs "
            f"{elem2}({tensor2.device})\n"
        )
    diff_str += "\n...\n\n"
    diff_str += f"{format_tensor(tensor1)}\n{tensor1}\n"
    diff_str += f"{format_tensor(tensor2)}\n{tensor2}\n"

    return diff_str


def get_op_name(op_func):
    """
    Extracts a simplified operation name from a function, trimming common prefixes and suffixes.

    Parameters:
    - op_func: The operation function.

    Returns:
    - A string representing the simplified name of the operation.
    """
    full_op_name = f"{op_func.__module__}.{op_func.__name__}"
    full_op_name = full_op_name.replace("torch._ops", "torch.ops")
    full_op_name = full_op_name.replace(".default", "")
    return full_op_name


def get_full_op_name(op_name):
    """
    Generates the full name of an operation including its module hierarchy
    and whether it's a forward or backward operation.

    Parameters:
    - op_name: The name of the operation.

    Returns:
    - The full name of the operation, considering its module hierarchy
    and operation type (forward or backward).
    """
    module_info = ModuleInfo(name=op_name, father=current_module_info, is_leaf=True)
    full_op_name = module_info.full_name()
    current_module_info.children.append(module_info)
    return full_op_name


def recursive_compare(out1, out2, atol, rtol, depth=0):
    """
    Recursively compares two outputs (tensors or collections of tensors)
    to check if they are approximately equal
    within specified absolute (atol) and relative (rtol) tolerances.

    Parameters:
    - out1, out2: The outputs to compare. These can be tensors,
    lists/tuples of tensors, or nested structures thereof.
    - atol (float): Absolute tolerance.
    - rtol (float): Relative tolerance.
    - depth (int): Recursion depth, used internally to format
    the output string with indentation for readability.

    Returns:
    - (bool, str): A tuple containing a boolean indicating whether
    the outputs are approximately equal and a string
    detailing differences if any.
    """
    indent = "    " * depth  # Indentation based on recursion depth
    tensors_diff_str = ""
    # Direct comparison for tensor objects
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        # Check if tensors are approximately equal considering atol and rtol
        if not torch.allclose(
            out1.to(out2.device).to(out2.dtype),
            out2,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        ):
            # Record difference
            tensors_diff_str += indent + "Tensor values are not close\n"
            diff_str = print_tensors_diff(out1, out2, atol, rtol)
            # Indent each line of the diff_str
            indented_diff_str = "\n".join(indent + line for line in diff_str.split("\n"))
            tensors_diff_str += indented_diff_str
            return False, tensors_diff_str
        return True, tensors_diff_str

    # Recursive comparison for list or tuple containers
    if isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
        all_results = []
        for i, (value1, value2) in enumerate(zip(out1, out2)):
            # Recurse into elements
            result, tensors_diff_str_child = recursive_compare(value1, value2, atol, rtol, depth + 1)
            all_results.append(result)
            if not result:
                tensors_diff_str += indent + f"Output {i} is not close:\n"
                tensors_diff_str += tensors_diff_str_child
        return all(all_results), tensors_diff_str

    # Check for NaN equivalence separately since NaN != NaN
    def are_both_nan(a, b):
        try:
            tensor_a, tensor_b = torch.tensor(a), torch.tensor(b)
        except TypeError:
            return False
        return torch.isnan(tensor_a.detach()).all() and torch.isnan(tensor_b.detach()).all()

    # Fallback comparison for non-tensor types
    if out1 != out2 and not are_both_nan(out1, out2):
        return False, indent + f"Values are not equal: {out1} vs {out2}\n"
    return True, tensors_diff_str


def recursive_print(args, top_level=True):
    """
    Recursively prints the structure and contents of the argument which can be a tensor,
    list, tuple, dictionary, or nested combinations thereof.

    Parameters:
    - args: The input to print, can be of any type but primarily aimed at tensors and collections.
    - top_level (bool): Indicates if the current call is at the top level (for pretty printing).
                        Top level entries are printed with an index or key, followed by a newline.

    Returns:
    - str: A string representation of the input, formatted for readability.
    """
    out_str = ""
    if isinstance(args, torch.Tensor):
        out_str += format_tensor(args)  # Format tensor for printing
    elif isinstance(args, (list, tuple)):
        # Recursively process list or tuple elements
        for i, x in enumerate(args):
            out_str += f"{i}: {recursive_print(x, False)}, \n" if top_level else f"{recursive_print(x, False)}, "
        # Enclose in brackets or parentheses based on type
        if not top_level:
            out_str = f"[{out_str}]" if isinstance(args, list) else f"({out_str})"
    elif isinstance(args, dict):
        # Recursively process dictionary key-value pairs
        for k, v in args.items():
            out_str += f"{k}: {recursive_print(v, False)}, \n"
        if not top_level:
            out_str = f"{{{out_str}}}"  # Enclose in braces
    else:
        # For non-collection types, simply convert to string
        out_str += str(args)

    # Print the formatted string if at the top level
    if top_level:
        print(out_str)

    return out_str


def compare_for_single_op(inputs_data_save_path, op_func, atol, rtol):
    """
    Compare the output of a single operation with saved data for correctness.

    Parameters:
    inputs_data_save_path (str): Path to the file containing saved input data.
    op_func (function): The operation function to test.
    atol (float): Absolute tolerance for comparison.
    rtol (float): Relative tolerance for comparison.

    Returns:
    tuple: The output of the operation and a boolean indicating if it passes the comparison.
    """
    # Load saved input data
    with open(inputs_data_save_path, "rb") as f:
        inputs_data = pickle.load(f)
        args_cpu = inputs_data["args"]
        kwargs_cpu = inputs_data["kwargs"]

    # Convert inputs to a specific device/format, if necessary
    # args_npu = convert_to_npu(args_cpu)
    # kwargs_npu = convert_to_npu(kwargs_cpu)

    args_npu = convert_to_gpu(args_cpu)
    kwargs_npu = convert_to_gpu(kwargs_cpu)

    # Compare the operation output with the expected results
    out, correct = compare_for_single_func(op_func, args_npu, kwargs_npu, atol, rtol, verbose=True)
    return correct, args_npu, kwargs_npu, out


def compare_for_single_func(func, args, kwargs, atol, rtol, func_name=None, verbose=True):
    """
    Compares the output of a function against expected results with given tolerances.

    Parameters:
    func (function): The function to test.
    args (tuple): Arguments for the function.
    kwargs (dict): Keyword arguments for the function.
    atol (float): Absolute tolerance.
    rtol (float): Relative tolerance.
    func_name (str, optional): The name of the function for logging. Defaults to None.
    verbose (bool, optional): Enables detailed logging. Defaults to False.

    Returns:
    tuple: The output of the function and a boolean indicating if it passes the comparison.
    """
    if func_name is None:
        func_name = get_op_name(func)
    print(f"{func_name} starts to run ...")

    # Convert arguments for CPU execution
    args_cpu = convert_to_cpu(args)
    kwargs_cpu = convert_to_cpu(kwargs)

    # Execute the function
    out = func(*args, **kwargs)
    correct = False

    try:
        try:
            # Attempt to run the function with CPU converted arguments
            out_cpu = func(*args_cpu, **kwargs_cpu)
        except RuntimeError as excp:
            # Handle runtime errors, possibly due to datatype issues
            print(excp)
            print("Convert to float32 ...")
            args_cpu = convert_to_dtype(args_cpu, torch.float32)
            kwargs_cpu = convert_to_dtype(kwargs_cpu, torch.float32)
            out_cpu = func(*args_cpu, **kwargs_cpu)

        # Compare the outputs
        correct, tensors_diff_str = recursive_compare(out, out_cpu, atol, rtol)
        if correct:
            print(f"{func_name} succeeds to pass CompareWithCPU test")
        else:
            print("\n============================")
            print(f"[ERROR] {func_name} fails to pass CompareWithCPU test")
            if verbose:
                # Log inputs and outputs for detailed debugging
                print("....... input .........")
                recursive_print(args)
                recursive_print(kwargs)
                print("...... output ........")
                recursive_print(out)
            print("\n...... compare with cpu .......")
            print(tensors_diff_str)
            print("============================")
    except Exception as excp:
        # Catch all other exceptions
        print(excp)
        print(f"[WARNING] {func_name} has not been tested!")

    return out, correct


def nan_inf_track_for_single_op(inputs_data_save_path, op_func):
    """
    Track NaN or Inf values in the output of a single operation using saved data.

    Parameters:
    inputs_data_save_path (str): Path to the file containing saved input data.
    op_func (function): The operation function to test.

    Returns:
    tuple: The output of the operation and a boolean indicating if NaN or Inf is detected.
    """
    # Load saved input data
    with open(inputs_data_save_path, "rb") as f:
        inputs_data = pickle.load(f)
        args_cpu = inputs_data["args"]
        kwargs_cpu = inputs_data["kwargs"]

    # Convert inputs for specific device/format, if necessary
    # args_npu = convert_to_npu(args_cpu)
    # kwargs_npu = convert_to_npu(kwargs_cpu)

    args_npu = convert_to_gpu(args_cpu)
    kwargs_npu = convert_to_gpu(kwargs_cpu)

    # Track NaN/Inf in the function output
    out, has_nan_or_inf = nan_inf_track_for_single_func(op_func, args_npu, kwargs_npu)
    return has_nan_or_inf, args_npu, kwargs_npu, out


def nan_inf_track_for_single_func(func, args, kwargs, func_name=None):
    """
    Detects NaN or Inf values in the output of a function.

    Parameters:
    func (function): The function to test.
    args (tuple): Arguments for the function.
    kwargs (dict): Keyword arguments for the function.
    func_name (str, optional): The name of the function for logging. Defaults to None.

    Returns:
    tuple: The output of the function and a boolean indicating if NaN or Inf is detected.
    """
    if func_name is None:
        func_name = get_op_name(func)
    print_str = ""
    print("\n============================")
    print(func_name)
    print("....... input .........")
    print_str += recursive_print(args)
    print_str += recursive_print(kwargs)
    out = func(*args, **kwargs)
    print("...... output ........")
    print_str += recursive_print(out)
    has_nan_or_inf = "[WARNING]" in print_str

    return out, has_nan_or_inf


def save_data_for_op(out, args, kwargs, save_dir, op_name, file_suffix=""):
    """
    Saves input and output data for an operation to files for later comparison or analysis.

    Parameters:
    out (Tensor): The output of the operation.
    args (tuple): Arguments used for the operation.
    kwargs (dict): Keyword arguments used for the operation.
    save_dir (str): The directory where data will be saved.
    op_name (str): The name of the operation.
    file_suffix (str, optional): A suffix for the filename. Defaults to ''.

    Returns:
    None
    """
    inputs_pkl_save_path = os.path.join(save_dir, f"{op_name}_inputs{file_suffix}.pkl")
    outputs_pkl_save_path = os.path.join(save_dir, f"{op_name}_outputs{file_suffix}.pkl")

    # Prepare data for saving
    inputs_data = {"args": convert_to_cpu(args), "kwargs": convert_to_cpu(kwargs)}
    outputs_data = convert_to_cpu(out)

    # Save data to files
    with open(inputs_pkl_save_path, "wb") as f:
        pickle.dump(inputs_data, f)
    with open(outputs_pkl_save_path, "wb") as f:
        pickle.dump(outputs_data, f)

    print(f"Input data is saved to {inputs_pkl_save_path},\n" f"output data is saved to {outputs_pkl_save_path}")


class LogToFile:
    """
    A context manager for redirecting stdout to a file.

    This class provides a mechanism for capturing the standard output
    to a specified file. It's useful for logging purposes, where the output
    of a block of code needs to be saved. If the file has not been written to
    before, it opens in write mode and logs the current timestamp. Otherwise,
    it appends to the existing file.

    Attributes:
        _has_cleared_files (dict):
            Tracks whether files have been cleared to avoid duplicate headers.
        filepath (str or None): Path to the log file. If None, stdout is not redirected.
        original_stdout (io.TextIOWrapper): Reference to the original stdout.

    """

    _has_cleared_files = {}  # Used to track whether each file has been cleared

    def __init__(self, filepath=None):
        """
        Initializes the context manager with the path to the log file.

        Args:
            filepath (str, optional): The path to the file where stdout will be redirected.
        """
        self.filepath = filepath
        self.original_stdout = sys.stdout

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        The stdout is redirected to the specified file. If the file has not been
        written to before, it is cleared and initialized with a timestamp.

        Returns:
            LogToFile: The runtime context object.
        """
        if self.filepath:
            # Check if this file has already been cleared
            if self.filepath not in self._has_cleared_files:
                # If not, open in "w" mode to clear it and mark as cleared
                self.file = open(self.filepath, "w", encoding="utf-8")
                self.file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + "\n")
                self._has_cleared_files[self.filepath] = True
            else:
                # If already cleared, open in "a" mode to append content
                self.file = open(self.filepath, "a", encoding="utf-8")
            sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context and restores the original stdout.

        The log file is closed if it was opened, and stdout is restored
        to its original state.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Traceback object.
        """
        sys.stdout = self.original_stdout
        if self.filepath:
            self.file.close()


# https://github.com/pytorch/pytorch/issues/94403
class TorchFuncMockNoDispatch:
    """
    Wraps a method to call it without the custom
    pytorch dispatcher
    """

    def __init__(self, pt_impl):
        self.pt_impl = pt_impl

    def __get__(self, obj, c):
        return partial(self, obj)

    def __call__(self, obj, *args, **kwargs):
        with _pop_mode_temporarily():
            return self.pt_impl(obj, *args, **kwargs)


class CompareWithCPU(TorchDispatchMode):
    """
    A class for comparing the outputs of tensor operations against
    their CPU results to ensure correctness.
    This is useful for debugging and verifying the consistency
    of operations across different devices.

    Attributes:
        enabled (bool): Flag to enable/disable comparison.
        atol (float): Absolute tolerance for comparison.
        rtol (float): Relative tolerance for comparison.
        target_list (list[str]):
            Specific operations to compare; if not empty, only these are considered.
        white_list (list[str]):
            Operations to ignore during comparison; considered if target_list is empty.
        dump_error_data (bool): If True, saves args of the first failing op and exits.
        verbose (bool): If True, prints detailed info about the args of the ops being compared.
        enable_ranks (list[int]):
            MPI ranks that are allowed to perform comparisons; None means all ranks.
        should_log_to_file (bool): If True, logs comparison results to a file.
        output_dir (str): Directory to save logs and error data.
        start_step (int): Step number to start comparisons.
        end_step (int): Step number to end comparisons.
    """

    TENSOR_FUNCS_NO_DISPATCH = [
        # Can't convert Stream argument to Python object
        "record_stream"
    ]

    def __enter__(self) -> None:
        self._pt_impls = {}
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __init__(
        self,
        enabled=True,
        atol=0.001,
        rtol=0.001,
        target_list=None,
        white_list=None,
        dump_error_data=False,
        verbose=True,
        enable_ranks=None,
        should_log_to_file=False,
        output_dir="",
        start_step=None,
        end_step=None,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.atol = atol
        self.rtol = rtol
        # Default whitelist has ops often non-deterministic or not key for comparison
        self.white_list = [
            "torch.ops._record_function_enter_new",
            "torch.ops.detach",
            "torch.ops.aten.detach",
            "torch.ops.uniform_",
            "torch.ops.set_.source_Storage",
            "torch.ops.set_.source_Storage_storage_offset",
            "torch.ops.new_empty",
            "torch.ops.aten.random_",
            "torch.ops.isinf",
            "torch.ops.aten.isinf",
            "torch.ops.aten.isnan",
            "torch.ops.isnan",
            "torch.ops.normal_",
            "torch.ops.barrier",
            "torch.ops.aten.randperm.generator",
            "torch.ops._to_copy",
            "torch.ops.aten.clone",
            "torch.ops.aten.t",
            "torch.ops.aten.empty.memory_format",
            "torch.ops.aten.empty_like",
            "torch.ops.profiler._record_function_enter_new",
            "torch.ops.profiler._record_function_exit._RecordFunction",
            "torch.ops.aten._unsafe_view",
            "torch.ops.aten.view",
            "torch.ops.aten._to_copy",
            "torch.ops.aten.copy_",
        ]
        if white_list is not None:
            self.white_list += white_list
        self.target_list = target_list if target_list is not None else []
        self.dump_error_data = dump_error_data
        self.verbose = verbose
        self.enable_ranks = enable_ranks
        self.should_log_to_file = should_log_to_file
        self.output_dir = output_dir
        self.step_cnt = 0
        self.start_step = start_step
        self.end_step = end_step
        self.is_active = True  # Initially active, can be toggled based on step counts
        self.update_active_state()
        self.global_rank = int(os.environ.get("RANK", "-1"))  # Fetch MPI rank if available
        self.file_suffix = f"_rank{self.global_rank}" if self.global_rank >= 0 else ""
        self.log_file_path = os.path.join(self.output_dir, f"compare_result{self.file_suffix}.txt")

    def update_active_state(self):
        """
        Updates the active state based on current step count and the start/end steps defined.
        """
        is_after_start = self.start_step is None or self.start_step <= self.step_cnt
        is_before_end = self.end_step is None or self.end_step > self.step_cnt
        self.is_active = is_after_start and is_before_end

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        The core method that intercepts tensor operations. It compares the output of each operation
        executed with its CPU counterpart, based on
        the specified tolerances, white list, and target list.
        """
        with torch._C.DisableTorchFunction():
            # Skip comparison if disabled or not active
            if not self.enabled or not self.is_active:
                return func(*args, **kwargs)

            op_name = get_op_name(func)  # Utility function to extract operation name
            full_op_name = get_full_op_name(op_name)  # May include namespace

            # LogToFile context manager for optional file logging
            with LogToFile(self.log_file_path if self.should_log_to_file else None):
                # Skip comparison for certain conditions (ranks, target list, white list)
                if self.enable_ranks is not None and self.global_rank not in self.enable_ranks:
                    return func(*args, **kwargs)
                if len(self.target_list) > 0 and op_name not in self.target_list:
                    print(f"{full_op_name} is not in target_list, pass")
                    return func(*args, **kwargs)
                if op_name in self.white_list:
                    print(f"{full_op_name} is in white_list, pass")
                    return func(*args, **kwargs)

                # Perform the actual comparison
                out, correct = compare_for_single_func(
                    func,
                    args,
                    kwargs,
                    atol=self.atol,
                    rtol=self.rtol,
                    func_name=full_op_name,
                    verbose=self.verbose,
                )

                # Handle comparison failure
                if self.dump_error_data and not correct:
                    save_data_for_op(out, args, kwargs, self.output_dir, op_name, self.file_suffix)
                    raise Exception("CompareWithCPU Failed!")

        return out

    def step(self):
        """
        Increments the step count and updates the active state. This method should be called
        at the beginning or end of each step (iteration) to properly manage the comparison scope.
        """
        if not self.enabled:
            return
        self.step_cnt += 1
        self.update_active_state()
        # Optional logging for active steps
        if self.is_active:
            with LogToFile(self.log_file_path if self.should_log_to_file else None):
                print(f"-------------------------  step = {self.step_cnt}  ----------------------\n")


class NanInfTracker(TorchDispatchMode):
    """
    A class for tracking NaN (Not a Number) and Inf (Infinity) values in tensor operations.
    This helps in identifying operations that produce these values, which are often indicative
    of numerical instabilities in computations.

    Attributes:
        enabled (bool):
            Flag to enable/disable NaN/Inf tracking.
        target_list (list[str]):
            Specific operations to track; if not empty, only these are considered.
        white_list (list[str]):
            Operations to ignore during tracking; considered if target_list is empty.
        enable_ranks (list[int]):
            MPI ranks that are allowed to perform tracking; None means all ranks.
        should_log_to_file (bool):
            If True, logs tracking results to a file.
        output_dir (str):
            Directory to save logs and error data.
        dump_error_data (bool):
            If True, saves args of the first failing op and exits.
        start_step (int):
            Step number to start tracking.
        end_step (int):
            Step number to end tracking.
    """

    TENSOR_FUNCS_NO_DISPATCH = [
        # Can't convert Stream argument to Python object
        "record_stream"
    ]

    def __enter__(self) -> None:
        self._pt_impls = {}
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __init__(
        self,
        enabled=True,
        target_list=None,
        white_list=None,
        enable_ranks=None,
        should_log_to_file=False,
        output_dir="",
        dump_error_data=False,
        start_step=None,
        end_step=None,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.white_list = [
            "torch.ops._record_function_enter_new",
            "torch.ops.detach",
            "torch.ops.aten.detach",
            "torch.ops.set_.source_Storage",
            "torch.ops.set_.source_Storage_storage_offset",
            "torch.ops.isinf",
            "torch.ops.aten.isinf",
            "torch.ops.aten.isnan",
            "torch.ops.isnan",
            "torch.ops.barrier",
            "torch.ops.aten.randperm.generator",
            "torch.ops.aten.empty.memory_format",
            "torch.ops.profiler._record_function_enter_new",
            "torch.ops.profiler._record_function_exit._RecordFunction",
        ]
        if white_list is not None:
            self.white_list += white_list
        self.target_list = target_list if target_list is not None else []
        self.enable_ranks = enable_ranks
        self.should_log_to_file = should_log_to_file
        self.output_dir = output_dir
        self.dump_error_data = dump_error_data
        self.step_cnt = 0
        self.start_step = start_step
        self.end_step = end_step
        self.is_active = True  # Initially active, can be toggled based on step counts
        self.update_active_state()
        self.global_rank = int(os.environ.get("RANK", "-1"))  # Fetch MPI rank if available
        self.file_suffix = f"_rank{self.global_rank}" if self.global_rank >= 0 else ""
        self.log_file_path = os.path.join(self.output_dir, f"nan_inf_report{self.file_suffix}.txt")

    def update_active_state(self):
        """
        Updates the active state based on current step count and the start/end steps defined.
        """
        is_after_start = self.start_step is None or self.start_step <= self.step_cnt
        is_before_end = self.end_step is None or self.end_step > self.step_cnt
        self.is_active = is_after_start and is_before_end

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        The core method that intercepts tensor operations. It checks the output of each operation
        for NaN or Inf values based on the specified white list and target list.
        """
        with torch._C.DisableTorchFunction():
            if not self.enabled or not self.is_active:
                return func(*args, **kwargs)

            op_name = get_op_name(func)  # Utility function to extract operation name
            full_op_name = get_full_op_name(op_name)  # May include namespace

            with LogToFile(self.log_file_path if self.should_log_to_file else None):
                if self.enable_ranks is not None and self.global_rank not in self.enable_ranks:
                    return func(*args, **kwargs)
                if len(self.target_list) > 0 and op_name not in self.target_list:
                    print(f"{full_op_name} is not in target_list, pass")
                    return func(*args, **kwargs)
                if op_name in self.white_list:
                    print(f"{full_op_name} is in white_list, pass")
                    return func(*args, **kwargs)

                out, has_nan_or_inf = nan_inf_track_for_single_func(func, args, kwargs, full_op_name)

                if self.dump_error_data and has_nan_or_inf:
                    save_data_for_op(out, args, kwargs, self.output_dir, op_name, self.file_suffix)
                    raise Exception("Nan or Inf Detected!")

                return out

    def step(self):
        """
        Increments the step count and updates the active state. This method should be called
        at the beginning or end of each step (iteration) to properly manage the tracking scope.
        """
        if not self.enabled:
            return
        self.step_cnt += 1
        self.update_active_state()
        if self.is_active:
            with LogToFile(self.log_file_path if self.should_log_to_file else None):
                print(f"-------------------------  step = {self.step_cnt}  ----------------------\n")
