
E: Unable to locate package libboost1.55-all-dev
E: Couldn't find any package by glob 'libboost1.55-all-dev'
E: Couldn't find any package by regex 'libboost1.55-all-dev'
    sudo apt-get libboost-all-dev
    
CMake Warning at cmake/OpenCVPackaging.cmake:23 (message):
  CPACK_PACKAGE_VERSION does not match version provided by version.hpp
  header!
Call Stack (most recent call first):
  CMakeLists.txt:1105 (include)
CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
Please set them or make sure they are set and tested correctly in the CMake files:
opencv_dep_CUDA_nppi_LIBRARY
    1,cpu:install opencv with:    -D WITH_CUDA=OFF ;will solve it
    compile finished, but when extract flow will failed; 
    what():  /home/xiang/git/temporal-segment-networks/3rd-party/opencv-2.4.13/modules/dynamicuda/include/opencv2/dynamicuda/dynamicuda.hpp:84: error: (-216) The library is compiled without CUDA support in function setDevice
    
    2,GPU:
    原因解析：cuda9不再支持2.0架构
    A:https://blog.csdn.net/MyStylee/article/details/79035585  ##但是上面的博客是针对opencv3版本，如果是opencv2版本，只需要改（1）（2）（3）（5）条即可，第（4）条不用改。如果改了，会出现opencv_error。
    A:https://blog.csdn.net/u014613745/article/details/78310916
    找到FindCUDA.cmake文件:
    [1]find_cuda_helper_libs(nppi) 替换为
    find_cuda_helper_libs(nppial)
    find_cuda_helper_libs(nppicc)
    find_cuda_helper_libs(nppicom)
    find_cuda_helper_libs(nppidei)
    find_cuda_helper_libs(nppif)
    find_cuda_helper_libs(nppig)
    find_cuda_helper_libs(nppim)
    find_cuda_helper_libs(nppist)
    find_cuda_helper_libs(nppisu)
    find_cuda_helper_libs(nppitc)
    [2]set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppi_LIBRARY};${CUDA_npps_LIBRARY}") 替换为
    set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppial_LIBRARY};${CUDA_nppicc_LIBRARY};${CUDA_nppicom_LIBRARY};${CUDA_nppidei_LIBRARY};${CUDA_nppif_LIBRARY};${CUDA_nppig_LIBRARY};${CUDA_nppim_LIBRARY};${CUDA_nppist_LIBRARY};${CUDA_nppisu_LIBRARY};${CUDA_nppitc_LIBRARY};${CUDA_npps_LIBRARY}")
    [3]unset(CUDA_nppi_LIBRARY CACHE) 替换为
    unset(CUDA_nppial_LIBRARY CACHE)
    unset(CUDA_nppicc_LIBRARY CACHE)
    unset(CUDA_nppicom_LIBRARY CACHE)
    unset(CUDA_nppidei_LIBRARY CACHE)
    unset(CUDA_nppif_LIBRARY CACHE)
    unset(CUDA_nppig_LIBRARY CACHE)
    unset(CUDA_nppim_LIBRARY CACHE)
    unset(CUDA_nppist_LIBRARY CACHE)
    unset(CUDA_nppisu_LIBRARY CACHE)
    unset(CUDA_nppitc_LIBRARY CACHE)
    找到OpenCVDetectCUDA.cmake文件:
    [4] ...
      set(__cuda_arch_ptx "")
      if(CUDA_GENERATION STREQUAL "Fermi")
        set(__cuda_arch_bin "2.0")
      elseif(CUDA_GENERATION STREQUAL "Kepler")
        set(__cuda_arch_bin "3.0 3.5 3.7")
      ...
    替换为
        ...
      set(__cuda_arch_ptx "")
      if(CUDA_GENERATION STREQUAL "Kepler")
        set(__cuda_arch_bin "3.0 3.5 3.7")
      elseif(CUDA_GENERATION STREQUAL "Maxwell")
        set(__cuda_arch_bin "5.0 5.2")
      ...
    [5]cuda9中有一个单独的halffloat(cuda_fp16.h)头文件,也应该被包括在opencv的目录里将头文件cuda_fp16.h添加至 opencv\modules\cudev\include\opencv2\cudev\common.hpp
    即在common.hpp中添加:
    #include <cuda_fp16.h>

nvcc fatal : Unsupported gpu architecture 'compute_20'
    问题原因：cuda9不再支持2.0架构,在9中原有的nppi被分解为多个库，查看cuda的bin文件夹发现多了以下等名称的以nppi为前缀的dll文件。
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D CUDA_GENERATION=Kepler .. 
    即上面的第（4）步，可修改或不修改

denseflow安装：https://blog.csdn.net/liuxiao214/article/details/85201268
/usr/bin/ld: 找不到 -lopencv_dep_nppial
/usr/bin/ld: 找不到 -lopencv_dep_nppicc
/usr/bin/ld: 找不到 -lopencv_dep_nppicom
/usr/bin/ld: 找不到 -lopencv_dep_nppidei
/usr/bin/ld: 找不到 -lopencv_dep_nppif
/usr/bin/ld: 找不到 -lopencv_dep_nppig
/usr/bin/ld: 找不到 -lopencv_dep_nppim
/usr/bin/ld: 找不到 -lopencv_dep_nppist
/usr/bin/ld: 找不到 -lopencv_dep_nppisu
/usr/bin/ld: 找不到 -lopencv_dep_nppitc
    （1）这些问题都是因为找不到相应的lib文件，locate libopencv_dep_nppial确实没有，但是有nppial，需要建立软连接libopencv_dep_nppial到/usr/local/lib目录下面，一定要到这个目录下才能找到
    （2）或者：
    mkdir build 
    cd build
    cmake .. 
    之后，build路径下生成了新的文件，分别到以下4个路径的link.txt文件中进行修改，
    cd dense_flow/build/CMakeFiles/extract_cpu.dir 
    cd dense_flow/build/CMakeFiles/extract_gpu.dir 
    cd dense_flow/build/CMakeFiles/extract_warp_gpu.dir 
    cd dense_flow/build/CMakeFiles/pydenseflow.dir
    将link.txt文件中的opencv_dep_nppi修改为nppi

caffe 安装
Could NOT find Doxygen (missing:  DOXYGEN_EXECUTABLE)
    apt install doxygen
    
-- Unable to determine MPI from MPI driver /usr/lib/openmpi/bin/mpicxx
-- Could NOT find MPI_CXX (missing:  MPI_CXX_LIBRARIES MPI_CXX_INCLUDE_PATH)
    1，sudo apt-get intall mpich
    或下载安装
    https://blog.csdn.net/qq_30239975/article/details/77703321
    https://www.cnblogs.com/xingkongyihao/p/9733260.html
    记得到lib/caffe里面删失败的build ；或在bash里面修改mkdir

python tools/build_of.py：    
OpenCV Error: Gpu API call (no kernel image is available for execution on the device) in call, file /home/xiang/git/temporal-segment-networks/3rd-party/opencv-2.4.13/modules/gpu/include/opencv2/gpu/device/detail/transform_detail.hpp, line 361
terminate called after throwing an instance of 'cv::Exception'
  what():  /home/xiang/git/temporal-segment-networks/3rd-party/opencv-2.4.13/modules/gpu/include/opencv2/gpu/device/detail/transform_detail.hpp:361: error: (-217) no kernel image is available for execution on the device in function call
Aborted (core dumped)
    未解决，网上说是gencode版本不对，
    直接使用cpu—extract
    
    

