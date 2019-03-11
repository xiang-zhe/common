########  更新openpose安装
0, git clone --recursive https://github.com/CMU-Perceptual-Computing-Lab/openpose
1, 
1.1, bash ./ubuntu/install_cmake.sh(cuda，cudnn)
1.2, apt-get install libopencv-dev(opencv)
2, 
2.1, Ubuntu - Anaconda should not be installed on your system. Anaconda includes a Protobuf version that is incompatible with Caffe.
2.2, apt-get install cmake-qt-gui(见官方文档)
2.2, 或者cd build 
	SCENARIO 1 -- Caffe not installed and OpenCV installed using apt-get
	make ..
	SCENARIO 2 -- Caffe installed and OpenCV build from source
	cmake -DOpenCV_INCLUDE_DIRS=/home/"${USER}"/softwares/opencv/build/install/include \
	-DOpenCV_LIBS_DIR=/home/"${USER}"/softwares/opencv/build/install/lib \
	-DCaffe_INCLUDE_DIRS=/home/"${USER}"/softwares/caffe/build/install/include \
	-DCaffe_LIBS=/home/"${USER}"/softwares/caffe/build/install/lib/libcaffe.so -DBUILD_CAFFE=OFF ..
	SCENARIO 3 -- OpenCV already installed
	cmake -DOpenCV_INCLUDE_DIRS=/home/"${USER}"/softwares/opencv/build/install/include \
	-DOpenCV_LIBS_DIR=/home/"${USER}"/softwares/opencv/build/install/lib ..
3, cd build/
make -j`nproc`

########  GPU openpose 安装（old）
0,确认显卡驱动已装好，

1,git clone --recursive https://github.com/CMU-Perceptual-Computing-Lab/openpose

2,cd /openpose/3rdparty
git clone https://github.com/CMU-Perceptual-Computing-Lab/caffe
#这一步是安装caffe框架的，如果没有做，在cmake-gui中configure时，会出错，也可以出错后再操作

3,sudo apt-get install cmake-qt-gui

4,sudo bash ./ubuntu/install_cmake.sh
#这一步会自动下载安装cuda，cuDNN，

5,OpenPose Configuration
#这一步见官网，即打开cmake-gui，指定源文件，和build文件，点击configure，确认，默认原生态编译即可，configure done后标红色的项目是可配置的，然后generate
  
  make的时候可以选择BUILD_PYTHON ON，python API可用，build/examples/tutorial_python
  https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md
  其中python的路径需要加入环境变量，否则找不到openpose库，指定else: sys.path.append('/usr/bin/python')
  指定sys.path.append('/home/ubuntu/openpose/build/python/openpose') ——_openpose.so的路径


6，cd build
make -j`nproc`


####一般出问题的过程好像是caffe的安装

附加：
如果使用指定cuda和cudnn，则需要先安装，如cuda9.0和cudnn7.1，
下载好cuda9.0和cudnn7.1，安装：
cuda安装过程中driver可以选择no（如果已经安装好了）

安装完后添加环境变量，在profile文件后面添加如下两行：
sudo vim /etc/profile
	export PATH=/usr/local/cuda-8.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source /etc/profile


cudnn下载developer版本：cuDNN v7.1.4 Developer Library for Ubuntu16.04 (Deb)

这runtime版本是不带头文件的，configure的时候会找不到cudnn，cuDNN v7.1.4 Runtime Library for Ubuntu16.04 (Deb)

关于报错：
F0829 09:36:36.046618 26766 cudnn.hpp:128] Check failed: status == CUDNN_STATUS_SUCCESS (3 vs. 0)  CUDNN_STATUS_BAD_PARAM
*** Check failure stack trace: ***

    @     0x7f756ac515cd  google::LogMessage::Fail()
    
    @     0x7f756ac53433  google::LogMessage::SendToLog()
    
    @     0x7f756ac5115b  google::LogMessage::Flush()
    
    @     0x7f756ac53e1e  google::LogMessageFatal::~LogMessageFatal()
    
    @     0x7f756a5b0bf8  caffe::CuDNNConvolutionLayer<>::Reshape()
    
    @     0x7f756a5df266  caffe::Net<>::Init()
    
    @     0x7f756a5e23aa  caffe::Net<>::Net()
    
    @     0x7f756d9a4d58  op::NetCaffe::initializationOnThread()
    
    @     0x7f756d9e5e2d  op::addCaffeNetOnThread()
    
    @     0x7f756d9e68e8  op::PoseExtractorCaffe::netInitializationOnThread()
    
    @     0x7f756d9ea980  op::PoseExtractorNet::initializationOnThread()
    
    @     0x7f756d9e26f1  op::PoseExtractor::initializationOnThread()
    
    @     0x7f756d9dd911  op::WPoseExtractor<>::initializationOnThread()
    
    @     0x7f756d89c301  op::SubThread<>::initializationOnThread()
    
    @     0x7f756d8a0fa8  op::Thread<>::initializationOnThread()
    
    @     0x7f756d8a11ad  op::Thread<>::threadFunction()
    
    @     0x7f756bb36c80  (unknown)
    
    @     0x7f756b2886ba  start_thread
    
    @     0x7f756b5a541d  clone
    
    @              (nil)  (unknown)
    
Aborted (core dumped)

这可能是由于多个cudnn版本冲突导致的，先卸载cudnn，/usr/bin/cudnn.h，再重新安装cuda9.0和cudnn7.1即可
