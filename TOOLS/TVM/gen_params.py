import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd

tarmet = 'cuda'
#use_cuda = 1
#use_llvm-mcpu = 0
#use_maligpu = 0
#use_39 = 0

IRmet = 'relay' #the 2nd gen nnvm
#use_nnvm = 0
#use_relay = 1

prefix,epoch = "model",0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
image_size = (112, 112)
opt_level = 3
shape_dict = {'data': (1, 3, *image_size)}
# "target" means your target platform you want to compile.
if tarmet == 'cuda':
    target = tvm.target.cuda(1060)
elif target == 'llvm': # for cpu
    target = 'llvm'
elif tarmet == 'llvm-mcpu':
    target = tvm.target.create("llvm -mcpu=haswell")
    #target = tvm.target.create("llvm -mcpu=broadwell")
elif tarmet == 'maligpu':
    #https://docs.tvm.ai/tutorials/nnvm/deploy_model_on_mali_gpu.html#sphx-glr-tutorials-nnvm-deploy-model-on-mali-gpu-py
    target = tvm.target.cuda("llvm device=0") #
elif tarmet == '39':
    # Here is the setting for my rk3399 board. If you don't use rk3399, you can query your target triple by execute `gcc -v` on your board.
    target_host = "llvm -target=aarch64-linux-gnu"
    # set target as  `tvm.target.mali` instead of 'opencl' to enable optimization for mali
    target = tvm.target.mali()

if IRmet == 'nnvm':
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
    with nnvm.compiler.build_config(opt_level=opt_level):
       graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
elif IRmet == 'relay':   
    sym, params = tvm.relay.frontend.from_mxnet(sym, shape=shape_dict, dtype='float32', arg_params=arg_params, aux_params=aux_params)
    #sym func of graph, relay IR mainly optimize sym
    with tvm.relay.build_config(opt_level=opt_level):
        graph, lib, params = tvm.relay.build_module.build(sym, target, params=params)
   
lib.export_library("./deploy_{}_{}_lib.so".format(tarmet, IRmet))
print('lib export succeefully')
with open("./deploy_{}_{}_graph.json".format(tarmet, IRmet), "w") as fo:
    if IRmet == 'nnvm':
        fo.write(graph.json())
    elif IRmet == 'relay':
        fo.write(graph)
with open("./deploy_{}_{}_param.params".format(tarmet, IRmet), "wb") as fo:
    if IRmet == 'nnvm':
        fo.write(nnvm.compiler.save_param_dict(params))
    elif IRmet == 'relay':   
        fo.write(tvm.relay.save_param_dict(params))


