try:
    from pycuda import gpuarray
except ImportError:
    print("Not using the GPU")


class BenchScalarArgument:
    def __init__(self, name, dtype, generator, key=False, enable=True):
        self.name = name
        self.dtype = dtype
        self.generator = generator
        self.parent = None
        self.size = 1
        self.value = None
        self.key = key
        self.enable = enable

    def clear_value(self):
        self.value = None

    def generate(self, gpu=False):
        return self.dtype(self.generator.generate())

    def set_size(self, input_size_list, turn):
        pass


class BenchArrayArgument:
    def __init__(self, name, dtype, generator, index=0, key=False, enable=True):
        self.name = name
        self.dtype = dtype
        self.index = index
        self.generator = generator
        self.size = None
        self.key = key
        self.enable = enable

    def generate(self, gpu=False):
        if gpu:
            return gpuarray.to_gpu(self.generator.generate().astype(self.dtype))
        else:
            return self.generator.generate().astype(self.dtype)

    def set_size(self, input_size_list, turn):
        if isinstance(input_size_list[0], list):
            self.size = input_size_list[self.index][turn]
        else:
            self.size = input_size_list[turn]
