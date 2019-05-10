####  TensorRT 4.0.1.6 GA

import tensorrt as trt

'''
## 2.8Python初始化TensorRT
TensorRT的两种初始化方法（流程与C++一样）：
　　　‣创建IBuilder对象去优化网络（创建后可生成序列化文件）
　　　‣创建IRuntime对象去执行优化网络（从序列化文件导入） 
在任何一种情况下，都必须实现一个日志记录接口，TensorRT通过该接口打印错误，警告和信息性消息. 禁止了信息性消息，只打印警告性和错误性消息
　　　　　　'''
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)


## 2.9 Python创建网络定义
# caffe
datatype = trt.infer.DataType.FLOAT
MODEL_PROTOTXT = '/data/mnist/mnist.prototxt'
CAFFE_MODEL = '/data/mnist/mnist.caffemodel'
builder = trt.infer.create_infer_builder(G_LOGGER)
network = builder.create_network()
parser = parsers.caffeparser.create_caffe_parser()
blob_name_to_tensor = parser.parse(CAFFE_MODEL,MODEL_PROTOTXT,network,datatype)
# tensorflow
from tensorrt.parsers import uffparser
import uff
uff.from_tensorflow_frozen_model(frozen_file, ["fc2/Relu"])
parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (1, 28, 28), 0)
parser.register_output("fc2/Relu")
# 注：TensorRT默认输入张量是CHW，从TensorFlow（默认NHWC）导入时，确保输入张量也是CHW，如果不是先转换为CHW。 
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCHSIZE, MAX_WORKSPACE)
# ONNX
from tensorrt.parsers import onnxparser
apex = onnxparser.create_onnxconfig()
apex.set_model_filename("model_file_path") 
apex.set_model_dtype(trt.infer.DataType.FLOAT) 
apex.set_print_layer_info(True) # Optional debug option 
apex.report_parsing_info() # Optional debug option
apex.add_verbosity()
apex.reduce_verbosity()
apex.set_verbosity_level(3)
trt_parser = onnxparser.create_onnxparser(apex)
data_type = apex.get_model_dtype()
onnx_filename = apex.get_model_file_name()
trt_parser.parse(onnx_filename, data_type)
trt_parser.convert_to_trt_network()
# retrieve the network from the parser
trt_network = trt_parsr.get_trt_network()

## 2.10 使用Python API 创建网络
builder = trt.infer.create_infer_builder(G_LOGGER)
network = builder.create_network()
data = network.add_input(INPUT_LAYERS[0], dt, (1, INPUT_H, INPUT_W))
weight_map = trt.utils.load_weights(weights_file)
conv1 = network.add_convolution(scale1.get_output(0), 20, (5,5),
weight_map["conv1filter"], weight_map["conv1bias"])
conv1.set_stride((1,1))
#　　　注：传给TensorRT的权重保存在主机内存中 
pool1 = network.add_pooling(conv1.get_output(0), trt.infer.PoolingType.MAX,(2,2))
pool1.set_stride((2,2))
ip1 = network.add_fully_connected(pool2.get_output(0), 500, weight_map["ip1filter"], weight_map["ip1bias"])
relu1 = network.add_activation(ip1.get_output(0),trt.infer.ActivationType.RELU)
prob = network.add_softmax(ip2.get_output(0))
prob.get_output(0).set_name(OUTPUT_LAYERS[0])
network.mark_output(prob.get_output(0))
## 2.11 Python构建推理引擎
builder.set_max_batch_size(max_batch_size)
builder.set_max_workspace_size(1 << 20)
engine = builder.build_cuda_engine(network)
## 2.12 Python序列化模型
builder = trt.infer.create_infer_builder(G_LOGGER)
network = builder.create_network()
engine = builder.build_cuda_engine(network)
modelstream = engine.serialize()
engine.destroy()
builder.destroy()
runtime = trt.infer.create_infer_runtime(GLOGGER)
engine =runtime.deserialize_cuda_engine(modelstream.data(),modelstream.size(),None)
modelstream.destroy()
## 2.13 Python执行推理
context = engine.create_execution_context()
d_input = cuda.mem_alloc(insize)
d_output = cuda.mem_alloc()
bindings = [int(d_input),int(d_output)]
context.enqueue(batch_size,bindings,stream.handle,None)
cuda.memcpy_dtoh_async(output,d_output,stream)
stream.synchronize()
return output


