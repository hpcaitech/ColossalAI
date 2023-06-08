from .builder import Builder


class ElixirSimulatorBuilder(Builder):
    NAME = "elixir_simulator"
    PREBUILT_IMPORT_PATH = "colossalai._C.elixir_simulator"

    def __init__(self):
        super().__init__(name=ElixirSimulatorBuilder.NAME,
                         prebuilt_import_path=ElixirSimulatorBuilder.PREBUILT_IMPORT_PATH)
        self.version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.relative_to_abs_path('elixir/simulator.cpp'),
        ]
        return ret

    def include_dirs(self):
        return []

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_flags(self):
        return []

    def builder(self) -> 'CppExtension':
        """
        This function should return a CppExtension object.
        """
        from torch.utils.cpp_extension import CppExtension

        return CppExtension(name=self.prebuilt_import_path,
                            sources=self.strip_empty_entries(self.sources_files()),
                            extra_compile_args={
                                'cxx': self.strip_empty_entries(self.cxx_flags()),
                            })
