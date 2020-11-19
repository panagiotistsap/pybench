try:
    from pycuda import gpuarray
    try:
        import pycuda.autoinit
    except:
        print("Pycuda is already initialized")
except ImportError:
    print("Not using the GPU")


class BenchScalarArgument:
    def __init__(self, name, dtype, generator):
        self.name = name
        self.dtype = dtype
        self.generator = generator
        self.parent = None
        self.size = 1

    def generate(self):
        return self.dtype(self.generator.generate())

    def set_size(self, input_size_list, turn):
        pass


class BenchArrayArgument:
    def __init__(self, name, dtype, generator, index=0, gpu=False):
        self.name = name
        self.dtype = dtype
        self.index = index
        self.generator = generator
        self.gpu = gpu
        self.size = None

    def generate(self):
        if self.gpu:
            return gpuarray.to_gpu(self.generator.generate().astype(self.dtype))
        else:
            return self.generator.generate().astype(self.dtype)

    def set_size(self, input_size_list, turn):
        if isinstance(input_size_list[0], list):
            self.size = input_size_list[self.index][turn]
        else:
            self.size = input_size_list[turn]
