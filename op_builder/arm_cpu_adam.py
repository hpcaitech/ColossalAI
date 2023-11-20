from .builder import Builder


class ArmCPUAdamBuilder(Builder):
    NAME = "arm_cpu_adam"
    PREBUILT_IMPORT_PATH = "colossalai._C.arm_cpu_adam"
    ext_type = "cpu"

    def __init__(self):
        super().__init__(name=ArmCPUAdamBuilder.NAME, prebuilt_import_path=ArmCPUAdamBuilder.PREBUILT_IMPORT_PATH)
        self.version_dependent_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.csrc_abs_path("cpu_adam_arm.cpp"),
        ]
        return ret

    def include_dirs(self):
        return [self.csrc_abs_path("includes")]

    def cxx_flags(self):
        extra_cxx_flags = [
            "-std=c++14",
            "-std=c++17",
            "-g",
            "-Wno-reorder",
            "-fopenmp",
        ]
        return ["-O3"] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        return []
