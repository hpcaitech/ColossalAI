"""Check optional Apex probes without importing the GPU-dependent test modules.

Execute the actual importlib imports and module-level guard from each test file.
This lets the dependency-free CI helper suite catch unused-import autofixes that
silently turn an optional dependency probe into an unconditional True.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
OPTIMIZER = "tests/test_zero/test_gemini/test_optim.py"
CHECKPOINT = "tests/test_checkpoint_io/test_gemini_checkpoint_io.py"
GUARDS = {OPTIMIZER: "APEX_AMP_AVAILABLE", CHECKPOINT: "APEX_AVAILABLE"}


def read_module(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)


def probe(path, modules):
    tree = read_module(path)
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "importlib"]
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(isinstance(child, ast.Name) and child.id == GUARDS[path] for child in ast.walk(node))
    )
    code = compile(ast.Module(body=[*imports, guard], type_ignores=[]), path, "exec")

    def import_optional(name):
        if name not in modules:
            raise ModuleNotFoundError(f"No module named {name!r}")
        value = modules[name]
        if isinstance(value, Exception):
            raise value
        return value

    namespace = {}
    with patch("importlib.import_module", side_effect=import_optional) as importer:
        exec(code, namespace)
    return namespace[GUARDS[path]], [call.args[0] for call in importer.call_args_list]


def normalization_modules():
    return {
        "apex.normalization": SimpleNamespace(FusedLayerNorm=object, FusedRMSNorm=object),
        "fused_layer_norm_cuda": SimpleNamespace(),
    }


class OptionalApexGuardTests(unittest.TestCase):
    def test_missing_apex_is_unavailable(self):
        for path in GUARDS:
            with self.subTest(path=path):
                available, imports = probe(path, {})
                self.assertFalse(available)
                self.assertTrue(imports, "The guard must attempt an import, not assign a constant")

    def test_normalization_does_not_substitute_for_legacy_amp(self):
        available, imports = probe(OPTIMIZER, normalization_modules())
        self.assertFalse(available)
        self.assertEqual(imports, ["apex.amp"])

    def test_importable_legacy_amp_is_available(self):
        available, imports = probe(OPTIMIZER, {"apex.amp": SimpleNamespace()})
        self.assertTrue(available)
        self.assertEqual(imports, ["apex.amp"])

    def test_fused_normalization_does_not_require_legacy_amp(self):
        available, imports = probe(CHECKPOINT, normalization_modules())
        self.assertTrue(available)
        self.assertEqual(imports, ["apex.normalization", "fused_layer_norm_cuda"])

    def test_python_only_apex_does_not_enable_fused_kernels(self):
        modules = normalization_modules()
        del modules["fused_layer_norm_cuda"]
        available, imports = probe(CHECKPOINT, modules)
        self.assertFalse(available)
        self.assertIn("fused_layer_norm_cuda", imports)

    def test_both_normalization_classes_are_required(self):
        for missing in ("FusedLayerNorm", "FusedRMSNorm"):
            with self.subTest(missing=missing):
                modules = normalization_modules()
                delattr(modules["apex.normalization"], missing)
                self.assertFalse(probe(CHECKPOINT, modules)[0])

    def test_unloadable_cuda_extension_is_unavailable(self):
        modules = normalization_modules()
        modules["fused_layer_norm_cuda"] = ImportError("undefined symbol")
        self.assertFalse(probe(CHECKPOINT, modules)[0])

    def test_unexpected_initialization_errors_are_not_silenced(self):
        for path, module in ((OPTIMIZER, "apex.amp"), (CHECKPOINT, "apex.normalization")):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                probe(path, {module: RuntimeError("unexpected failure")})

    def test_optimizer_skip_uses_the_probe(self):
        function = next(
            node
            for node in read_module(OPTIMIZER).body
            if isinstance(node, ast.FunctionDef) and node.name == "test_optim"
        )
        marker = next(
            node
            for node in function.decorator_list
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "skipif"
        )
        condition = compile(ast.Expression(marker.args[0]), OPTIMIZER, "eval")
        for available in (True, False):
            with self.subTest(available=available):
                self.assertEqual(eval(condition, {GUARDS[OPTIMIZER]: available}), not available)

    def test_checkpoint_keeps_native_normalization_without_apex(self):
        expected = {"exam_state_dict_with_origin", "exam_state_dict"}
        functions = [
            node for node in read_module(CHECKPOINT).body if isinstance(node, ast.FunctionDef) and node.name in expected
        ]
        self.assertEqual({node.name for node in functions}, expected)
        for node in functions:
            setting = next(
                child
                for child in node.body
                if isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "enable_fused_normalization"
                    for target in child.targets
                )
            )
            condition = compile(ast.Expression(setting.value), CHECKPOINT, "eval")
            for tp_size in (1, 2):
                for available in (True, False):
                    with self.subTest(function=node.name, tp_size=tp_size, available=available):
                        self.assertEqual(
                            eval(condition, {GUARDS[CHECKPOINT]: available, "tp_size": tp_size}),
                            tp_size > 1 and available,
                        )


if __name__ == "__main__":
    unittest.main()
