import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import tvm.relay as relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import mxnet as mx
from mxnet import ndarray as nd
from sklearn import preprocessing
import time
import cv2
import os


image_size = (112, 112)
opt_level = 2
shape_dict = {'data': (1, 3, *image_size)}
#### DEVICE CONFIG ####
target = tvm.target.cuda()
#### TUNING OPTION ####
network = 'mxnet'
log_file = "%s.log" % network
dtype = 'float32'
tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'n_trial': 1,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}
def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 112, 112)
    output_shape = (batch_size, 512)
    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        net, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        net, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        '''
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = nnvm.frontend.from_mxnet(block)
        net = nnvm.sym.softmax(net)
        '''
        '''
        prefix,epoch = "model",0
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        block = sym
        net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype,)
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        '''
        prefix,epoch = "model",0
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        relay_func, relay_params = relay.frontend.from_mxnet(sym, shape_dict, arg_params=arg_params, aux_params=aux_params)
        net, params = relay_func, relay_params
    else:
        raise ValueError("Unsupported network: " + name)
    return net, params, input_shape, output_shape
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    net, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "./deploy_{}_{}_net.tar".format('tarmet', 'IRmet')
        #lib.export_library(tmp.relpath(filename))
        lib.export_library(filename)
        with open("./deploy_{}_{}_graph.json".format('tarmet', 'IRmet'), "w") as fo:
            fo.write(graph)
        with open("./deploy_{}_{}_param.params".format('tarmet', 'IRmet'), "wb") as fo:
            fo.write(tvm.relay.save_param_dict(params))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
tune_and_evaluate(tuning_option)

exit(1)
class FaceFeatures(object):
    def __init__(self):
        #ctx = tvm.cpu(0)
        ctx = tvm.gpu(0)        
        loaded_json = open("./deploy_gpu2_graph.json").read()
        loaded_lib = tvm.module.load("./deploy_gpu2_lib.tar")
        loaded_params = bytearray(open("./deploy_gpu2_param.params", "rb").read())
        self.module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        self.module.load_params(loaded_params)
    def get_feature(self,face_img=None):        
        input_data = tvm.nd.array(face_img)
        self.module.run(data=input_data)
        feature = self.module.get_output(0).asnumpy()
        feature = preprocessing.normalize(feature).flatten()
        return feature

data_shape = (1,3,112,112)
f = FaceFeatures()        
#input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
input_data = tvm.nd.array(np.zeros(data_shape).astype("float32"))
for i in range(100):
    f1 = f.get_feature(face_img=input_data)
start = time.time()
for i in range(1000):
    f1 = f.get_feature(face_img=input_data)
print(time.time()-start)

exit(0)
im = cv2.imread('/home/xiang/git/insightface/deploy/Tom_Hanks_54745.png')
im = im.transpose(2,0,1).astype("float32")#.reshape(1,3,112,112)
print(im.shape)
f1 = f.get_feature(face_img=im)
print(f1)


exit(0)
ctx = tvm.cpu()
# load the module back.
loaded_json = open("./deploy_graph.json").read()
loaded_lib = tvm.module.load("./deploy_lib.so")
loaded_params = bytearray(open("./deploy_param.params", "rb").read())


im = cv2.imread('/home/xiang/git/insightface/deploy/Tom_Hanks_54745.png')#.reshape(data_shape)
#im = im[...,::-1]
im = im.transpose(2,0,1)
#im = im.reshape(data_shape)
#print(im.shape)
print(im)
#exit(0)
'''
im = cv2.imread('../../deploy/2.jpeg')
im = cv2.resize(im, data_shape[2:])
#im = im.reshape((1,3,im.shape[0],im.shape[1]))
im = im.reshape(data_shape)
print(im.shape)'''

input_data = tvm.nd.array(im.astype("float32"))
#input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

# Tiny benchmark test.
import time
for i in range(100):
   t0 = time.time()
   module.run(data=input_data)
   f1 = module.get_output(0).asnumpy() ##[[]]
   f1 = preprocessing.normalize(f1).flatten() ##[]
   print(time.time() - t0)
print(f1)

#exit(0)
im = im/255
for i in range(100):
   t0 = time.time()
   module.run(data=input_data)
   f2 = module.get_output(0).asnumpy() ##[[]]
   f2 = preprocessing.normalize(f2).flatten() ##[]
   print(time.time() - t0)
print(f2)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
print(np.dot(f1, f1.T))




