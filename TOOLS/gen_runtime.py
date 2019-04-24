import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd
from sklearn import preprocessing

import cv2

class FaceFeatures(object):
    def __init__(self):
        ctx = tvm.cpu(0)
        loaded_json = open("./deploy_graph.json").read()
        loaded_lib = tvm.module.load("./deploy_lib.so")
        loaded_params = bytearray(open("./deploy_param.params", "rb").read())
        self.module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        self.module.load_params(loaded_params)
    def get_feature(self,face_img=None):        
        input_data = tvm.nd.array(face_img)
        self.module.run(data=input_data)
        feature = self.module.get_output(0).asnumpy()
        feature = preprocessing.normalize(feature).flatten()
        return feature

f = FaceFeatures()
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

data_shape = (1,3,112,112)
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




