try:
    from pycuda import gpuarray
    try:
        import pycuda.autoinit
    except:
        print("Pycuda is already initialized")
except:
    print("Cant use GPU")


class BenchScalarArgument:
    def __init__(self, name, dtype, generator):
        self.name = name
        self.dtype = dtype
        self.generator = generator
        self.parent = None
        self.size = 1

    def generate(self):
        return self.generator.generate()

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
            return gpuarray.to_gpu(self.generator.generate())
        else:
            return self.generator.generate()

    def set_size(self, input_size_list, turn):
        if isinstance(input_size_list[0], list):
            self.size = input_size_list[self.index][turn]
        else:
            self.size = input_size_list[turn]
