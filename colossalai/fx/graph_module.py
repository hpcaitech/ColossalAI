import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent

try:
    from torch.fx.graph import Graph, PythonCode, _PyTreeCodeGen
    from torch.fx.graph_module import GraphModule, _exec_with_source, _forward_from_src, _WrappedCall

    from colossalai.fx.codegen.activation_checkpoint_codegen import ActivationCheckpointCodeGen

    COLOGM = True
except:
    from torch.fx.graph import Graph
    from torch.fx.graph_module import GraphModule

    COLOGM = False

if COLOGM:

    class ColoGraphModule(GraphModule):
        def __init__(
            self,
            root: Union[torch.nn.Module, Dict[str, Any]],
            graph: Graph,
            class_name: str = "GraphModule",
            ckpt_codegen: bool = True,
        ):
            if ckpt_codegen:
                graph.set_codegen(ActivationCheckpointCodeGen())
            super().__init__(root, graph, class_name)

        def bind(self, ckpt_def, globals):
            """Bind function needed for correctly execute gm forward

            We need to bind checkpoint functions and saved_tensor_hooks functions
            to gm so that we could correctly execute gm forward

            Args:
                ckpt_def (_type_): definition before the forward function
                globals (_type_): global variables
            """

            ckpt_code = "\n".join(ckpt_def)
            globals_copy = globals.copy()
            _exec_with_source(ckpt_code, globals_copy)
            func_list = [func for func in globals_copy.keys() if "checkpoint" in func or "pack" in func]
            for func in func_list:
                tmp_func = globals_copy[func]
                setattr(self, func, tmp_func.__get__(self, self.__class__))
                del globals_copy[func]

        def recompile(self) -> PythonCode:
            """
            Recompile this GraphModule from its ``graph`` attribute. This should be
            called after editing the contained ``graph``, otherwise the generated
            code of this ``GraphModule`` will be out of date.
            """
            if isinstance(self._graph._codegen, _PyTreeCodeGen):
                self._in_spec = self._graph._codegen.pytree_info.in_spec
                self._out_spec = self._graph._codegen.pytree_info.out_spec
            python_code = self._graph.python_code(root_module="self")
            self._code = python_code.src

            # To split ckpt functions code and forward code
            _code_list = self._code.split("\n")
            _fwd_def = [item for item in _code_list if "def forward" in item][0]
            _fwd_idx = _code_list.index(_fwd_def)
            ckpt_def = _code_list[:_fwd_idx]
            self._code = "\n".join(_code_list[_fwd_idx:])

            self.bind(ckpt_def, python_code.globals)

            cls = type(self)
            cls.forward = _forward_from_src(self._code, python_code.globals)

            # Determine whether this class explicitly defines a __call__ implementation
            # to wrap. If it does, save it in order to have wrapped_call invoke it.
            # If it does not, wrapped_call can use a dynamic call to super() instead.
            # In most cases, super().__call__ should be torch.nn.Module.__call__.
            # We do not want to hold a reference to Module.__call__ here; doing so will
            # bypass patching of torch.nn.Module.__call__ done while symbolic tracing.
            cls_call = cls.__call__ if "__call__" in vars(cls) else None

            if "_wrapped_call" not in vars(cls):
                cls._wrapped_call = _WrappedCall(cls, cls_call)  # type: ignore[attr-defined]

            def call_wrapped(self, *args, **kwargs):
                return self._wrapped_call(self, *args, **kwargs)

            cls.__call__ = call_wrapped

            # reset self._code to original src, otherwise to_folder will be wrong
            self._code = python_code.src
            return python_code

        def to_folder(self, folder: Union[str, os.PathLike], module_name: str = "FxModule"):
            """Dumps out module to ``folder`` with ``module_name`` so that it can be
            imported with ``from <folder> import <module_name>``

            Args:

                folder (Union[str, os.PathLike]): The folder to write the code out to

                module_name (str): Top-level name to use for the ``Module`` while
                    writing out the code
            """
            folder = Path(folder)
            Path(folder).mkdir(exist_ok=True)
            torch.save(self.state_dict(), folder / "state_dict.pt")
            tab = " " * 4

            # we add import colossalai here
            model_str = f"""
import torch
from torch.nn import *
import colossalai


class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

            def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
                safe_reprs = [
                    nn.Linear,
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ]
                if type(module) in safe_reprs:
                    return f"{module.__repr__()}"
                else:
                    return None

            blobified_modules = []
            for module_name, module in self.named_children():
                module_str = _gen_model_repr(module_name, module)
                if module_str is None:
                    module_file = folder / f"{module_name}.pt"
                    torch.save(module, module_file)
                    blobified_modules.append(module_name)
                    module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
                    module_str = f"torch.load(r'{module_file}') # {module_repr}"
                model_str += f"{tab*2}self.{module_name} = {module_str}\n"

            for buffer_name, buffer in self._buffers.items():
                if buffer is None:
                    continue
                model_str += f"{tab*2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"

            for param_name, param in self._parameters.items():
                if param is None:
                    continue
                model_str += f"{tab*2}self.{param_name} = torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype}))\n"

            model_str += f"{tab*2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
            model_str += f"{_addindent(self.code, 4)}\n"

            module_file = folder / "module.py"
            module_file.write_text(model_str)

            init_file = folder / "__init__.py"
            init_file.write_text("from .module import *")

            if len(blobified_modules) > 0:
                warnings.warn(
                    "Was not able to save the following children modules as reprs -"
                    f"saved as pickled files instead: {blobified_modules}"
                )

else:

    class ColoGraphModule(GraphModule):
        def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, class_name: str = "GraphModule"):
            super().__init__(root, graph, class_name)
