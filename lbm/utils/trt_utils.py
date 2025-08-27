'''
Adapted from https://github.com/lyuwenyu/RT-DETR
'''
from collections import namedtuple, OrderedDict
import time
import contextlib
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda 
import os


class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='cuda', max_point_num=100, max_memory_size=96, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_point_num = max_point_num
        self.max_memory_size = max_memory_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Create CUDA stream
        if self.backend == 'cuda':
            self.stream = cuda.Stream()
            
        # Get input and output names
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        # Initialize bindings
        self.bindings = self.get_bindings(self.engine, self.context, self.max_point_num, self.max_memory_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        
        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Engine file not found: {path}")
        try:
            trt.init_libnvinfer_plugins(self.logger, '')
            with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Failed to load engine: {str(e)}")
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_point_num=100, max_memory_size=96, device=None):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            # Handle dynamic shapes
            if shape[0] == -1:
                shape[0] = max_point_num
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            if len(shape) > 1 and shape[1] == -1:
                shape[1] = max_point_num
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)
            
            if len(shape) > 2 and shape[2] == -1:
                shape[2] = max_memory_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if n not in blob:
                raise ValueError(f"Missing input: {n}")
            if not isinstance(blob[n], torch.Tensor):
                raise TypeError(f"Input {n} must be torch.Tensor")

        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
                self.bindings_addr[n] = blob[n].data_ptr()

        for n in self.input_names:
            self.context.set_tensor_address(n, blob[n].data_ptr())

        self.context.execute_v3(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def async_run_cuda(self, blob):
        # Asynchronous inference
        for n in self.input_names:
            if n not in blob:
                raise ValueError(f"Missing input: {n}")
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        # Set tensor addresses
        for i, name in enumerate(self.engine):
            self.context.set_tensor_address(name, self.bindings_addr[name])
        
        # Execute asynchronous inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Get outputs
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        # Synchronize stream
        self.stream.synchronize()
        
        return outputs

    def __call__(self, blob):
        if self.backend == 'cuda':
            return self.async_run_cuda(blob)
        else:
            return self.run_torch(blob)

    def synchronize(self):
        if self.backend == 'cuda':
            self.stream.synchronize()
        elif self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 

    def __del__(self):
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.synchronize()
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()