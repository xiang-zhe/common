#### 

## 3.1. Importing TensorRT Into Python
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

## 3.2. Creating A Network Definition In Python
'''
The first step in performing inference with TensorRT is to create a TensorRT network from your model. 
The easiest way to achieve this is to import the model using the TensorRT parser library
An alternative is to define the model directly using the TensorRT Network API
'''
# 3.2.1. Creating A Network Definition From Scratch Using The Python API
'''
When creating a network, you must first define the engine and create a builder object for inference. 
The Python API is used to create a network and engine from the Network APIs. The network definition reference is used to add various layers to the network. 
The following code illustrates how to create a simple network with Input, Convolution, Pooling, FullyConnected, Activation and SoftMax layers. 
'''
# Create the builder and network
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
	# Configure the network layers based on the weights provided. In this case, the weights are imported from a pytorch model. 
	# Add an input layer. The name is a string, dtype is a TensorRT dtype, and the shape can be provided as either a list or tuple.
	input_tensor = network.add_input(name=INPUT_NAME, dtype=trt.float32, shape=INPUT_SHAPE)

	# Add a convolution layer
	conv1_w = weights['conv1.weight'].numpy()
	conv1_b = weights['conv1.bias'].numpy()
	conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
	conv1.stride = (1, 1)

	pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
	pool1.stride = (2, 2)
	conv2_w = weights['conv2.weight'].numpy()
	conv2_b = weights['conv2.bias'].numpy()
	conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
	conv2.stride = (1, 1)

	pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
	pool2.stride = (2, 2)

	fc1_w = weights['fc1.weight'].numpy()
	fc1_b = weights['fc1.bias'].numpy()
	fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

	relu1 = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)

	fc2_w = weights['fc2.weight'].numpy()
	fc2_b = weights['fc2.bias'].numpy()
	fc2 = network.add_fully_connected(relu1.get_output(0), OUTPUT_SIZE, fc2_w, fc2_b)

	fc2.get_output(0).name =OUTPUT_NAME
	network.mark_output(fc2.get_output(0))
	
# 3.2.2. Importing A Model Using A Parser In Python
'''
To import a model using a parser, you will need to perform the following high-level steps:

    Create the TensorRTbuilder and network.
    Create the TensorRT parser for the specific format.
    Use the parser to parse the imported model and populate the network.
The builder must be created before the network because it serves as a factory for the network. Different parsers have different mechanisms for marking network outputs. 
'''
# 3.2.3. Importing From Caffe Using Python
import tensorrt as trt
datatype = trt.float32  #Define the data type. In this example, we will use float32.
deploy_file = 'data/mnist/mnist.prototxt'
model_file = 'data/mnist/mnist.caffemodel'
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
    model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=datatype) 
    # The parser returns the model_tensors, which is a table containing the mapping from tensor names to ITensor objects. 
# 3.2.4. Importing From TensorFlow Using Python
'''
Create a frozen TensorFlow model for the tensorflow model. The instructions on freezing a TensorFlow model into a stream can be found in Freezing A TensorFlow Graph. 
Use the UFF converter to convert a frozen tensorflow model to a UFF file. Typically, this is as simple as:
    convert-to-uff frozen_inference_graph.pb    
'''
model_file = '/data/mnist/mnist.uff'
with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    	parser.register_input("Placeholder", (1, 28, 28))
    	parser.register_output("fc2/Relu")
parser.parse(model_file, network)
# 3.2.5. Importing From ONNX Using Python
with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open(model_path, 'rb') as model:
        parser.parse(model.read())
# 3.2.6. Importing From PyTorch And Other Frameworks  
'''
Using TensorRT with PyTorch (or any other framework with NumPy compatible weights) involves replicating the network architecture using the TensorRT API, (see Creating A Network Definition From Scratch Using The Python API), and then copying the weights from PyTorch.
'''

## 3.3. Building An Engine In Python
'''
One of the functions of the builder is to search through its catalog of CUDA kernels for the fastest implementation available, and thus it is necessary use the same GPU for building as that on which the optimized engine will run. 
Two particularly important properties are the maximum batch size and the maximum workspace size.
    The maximum batch size specifies the batch size for which TensorRT will optimize. At runtime, a smaller batch size may be chosen.
    Layer algorithms often require temporary workspace. This parameter limits the maximum size that any layer in the network can use. If insufficient scratch is provided, it is possible that TensorRT may not be able to find an implementation for a given layer.

'''
builder.max_batch_size = max_batch_size
builder.max_workspace_size = 1 <<  20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
with trt.Builder(TRT_LOGGER) as builder:
    with builder.build_cuda_engine(network) as engine:
        ######### Do inference here. ########

## 3.4. Serializing A Model In Python
serialized_engine = engine.serialize()
with trt.Runtime(TRT_LOGGER) as runtime: # Deserializing requires creation of a runtime object
    engine = runtime.deserialize_cuda_engine(serialized_engine)
with open(“sample.engine”, “wb”) as f:
	f.write(engine.serialize())
with open(“sample.engine”, “rb”) as f, trt.Runtime(TRT_LOGGER) as runtime:
	engine = runtime.deserialize_cuda_engine(f.read())

## 3.5. Performing Inference In Python
#1,Allocate some host and device buffers for inputs and outputs:
    h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)
    h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
#2,Create some space to store intermediate activation values. Since the engine holds the network definition and trained parameters, additional space is necessary. These are held in an execution context: 
with engine.create_execution_context() as context:
	# Transfer input data to the GPU.
	cuda.memcpy_htod_async(d_input, h_input, stream)
	# Run inference.
	context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
	# Transfer predictions back from the GPU.
	cuda.memcpy_dtoh_async(h_output, d_output, stream)
	# Synchronize the stream
	stream.synchronize()
	# Return the host output. 
    return h_output    

#### 4. Extending TensorRT With Custom Layers
## 4.2. Adding Custom Layers Using The Python API
# 4.2.1. Example 1: Adding A Custom Layer to a TensorRT Network Using Python
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            lrelu_slope_field = trt.PluginField("neg_slope", np.array([0.1], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            field_collection = trt.PluginFieldCollection([lrelu_slope_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin

def main():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 2**20
        input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 1))
        lrelu = network.add_plugin_v2(inputs=[input_layer], plugin=get_trt_plugin("LReLU_TRT"))
        lrelu.get_output(0).name = "outputs"
        network.mark_output(lrelu.get_output(0))
# 4.2.2. Example 2: Adding A Custom Layer That Is Not Supported In UFF Using Python
trt.init_libnvinfer_plugins(TRT_LOGGER, '') #(or load the .so file where you have registered your own plugin)
tf_sess = tf.InteractiveSession()
tf_input = tf.placeholder(tf.float32, name="placeholder")
tf_lrelu = tf.nn.leaky_relu(tf_input, alpha=lrelu_alpha, name="tf_lrelu")
tf_result = tf_sess.run(tf_lrelu, feed_dict={tf_input: lrelu_args})
tf_sess.close()
trt_lrelu = gs.create_plugin_node(name="trt_lrelu", op="LReLU_TRT", negSlope=lrelu_alpha)
namespace_plugin_map = {
            "tf_lrelu": trt_lrelu
 }
dynamic_graph = gs.DynamicGraph(tf_lrelu.graph)
dynamic_graph.collapse_namespaces(namespace_plugin_map)
uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), ["trt_lrelu"], output_filename=model_path, text=True)
parser = trt.UffParser()
parser.register_input("placeholder", [lrelu_args.size])
parser.register_output("trt_lrelu")
parser.parse(model_path, trt_network)
## 4.3. Using Custom Layers When Importing A Model From A Framework




    
