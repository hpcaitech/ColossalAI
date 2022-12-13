class PatchRegistry:

    def __init__(self, name):
        self.name = name
        self.store = {}

    def register(self, source):

        def wrapper(func):
            self.store[source] = func
            return func

        return wrapper

    def get(self, source):
        assert source in self.store
        target = self.store[source]
        return target

    def has(self, source):
        return source in self.store


meta_patched_function = PatchRegistry(name='patched_functions_for_meta_execution')
meta_patched_module = PatchRegistry(name='patched_modules_for_meta_execution')
bias_addition_function = PatchRegistry(name='patched_function_for_bias_addition')
bias_addition_module = PatchRegistry(name='patched_module_for_bias_addition')
bias_addition_method = PatchRegistry(name='patched_method_for_bias_addition')
