import mxnet as mx
from mxnet.gluon.model_zoo import vision
import time
import os
import tvm
import tvm.relay as relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import numpy as np
'''
#瓶颈是cuDNN的INT8的convolution太慢了，以后会考虑跟TVM结合，使用原生的高性能kernel。
Starting MXNet timed run
3.7011698950000005
Building TensorRT engine
Warming up TensorRT
Starting TensorRT timed run
3.563012433000001


batch_shape = (1, 3, 224, 224)

resnet18 = vision.resnet18_v2(pretrained=True)
resnet18.hybridize()
resnet18.forward(mx.nd.zeros(batch_shape))
resnet18.export('resnet18_v2')
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet18_v2', 0)
'''

batch_shape = (1, 3, 112, 112)
sym, arg_params, aux_params = mx.model.load_checkpoint('model', 0)
image_size = (112, 112)
opt_level = 3
shape_dict = {'data': (1, 3, *image_size)}

# create inputdata
inputdata = mx.nd.zeros(batch_shape)

# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
executor = sym.simple_bind(ctx=mx.gpu(0), data=batch_shape, grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)
# Warmup
print('Warming up MXNet')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=inputdata)
    y_gen[0].wait_to_read()
# Timing
print('Starting MXNet timed run')
start = time.process_time()
for i in range(0, 100):
    y_gen = executor.forward(is_train=False, data=inputdata)
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)


# execute with TensorRT
print('Building TensorRT engine')
os.environ['MXNET_USE_TENSORRT'] = '1'
arg_params.update(aux_params)
all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in arg_params.items()])
executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(0), all_params=all_params, data=batch_shape, grad_req='null', force_rebind=True)
# warmup
print('Warming up TensorRT')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=inputdata)
    y_gen[0].wait_to_read()
# Timing
print('Starting TensorRT timed run')
start = time.process_time()
for i in range(0, 100):
    y_gen = executor.forward(is_train=False, data=inputdata)
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)


# Execute with TVM

exit(0)
#very slow, why?
target = tvm.target.cuda(model = 1060)
sym, params = tvm.relay.frontend.from_mxnet(sym, shape=shape_dict, dtype="float32", arg_params=arg_params, aux_params=aux_params)
with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', sym, tvm.gpu(0), target)
func = intrp.evaluate(sym)
# Warmup
print('Warming up TVM')
inputdata = np.zeros(batch_shape)
for i in range(0, 10):
    output = func(tvm.nd.array(inputdata.astype("float32")), **params).asnumpy()
# Timing
print('Starting TVM timed run')
start = time.process_time()
for i in range(0, 100):
    output = func(tvm.nd.array(inputdata.astype('float32')), **params).asnumpy()
end = time.time()
print(time.process_time() - start)
