##打开android studio
    cd android-studio/bin/
    ./studio.sh
    

#### 最快方式是下载apk直接安装，http://download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk
    安装adb
    sudo  apt-get  install android-tools-adb
    链接手机debug模式
    adb devices
    安装apk
    adb install -r -g /home/xiang/Downloads/TfLiteCameraDemo.apk



####The simplest way to compile the demo app yourself, and try out changes to the project code is to use AndroidStudio.
    安装Android Studio（https://developer.android.google.cn/studio/）
        下载后解压，cd进入android-studio/bin目录 ./studio.sh 进行安装
            Error:Could not find com.android.tools.build:gradle:3.0.1
                修改如下添加maven...或者maven { url 'http://maven.aliyun.com/nexus/content/groups/public/' }
                buildscript {
                repositories {

                    jcenter()
                    maven {
                        url 'https://maven.google.com'
                    }
                }
            Error:Failed to find target with hash string 'android-23'
                Android Studio》file》setttings》system settings》android SDK》SDK platform》23
        	Error:Gradle sync failed: Failed to find Build Tools revision 26.0.2
                会出现提示没有改版本，弹出一个链接叫你去下载该版本
                
    Open an existing Android Studio project选择tensorflow/examples/android
    打开工程中android模块下的build.gradle文件，找到nativeBuildSystem变量并且设置它为none如果它还没设置。
        // set to 'bazel', 'cmake', 'makefile', 'none'
        def nativeBuildSystem = 'none'  
    点击run按钮或者使用Run-> Run 'android'从顶部菜单。如果它询问你使用Instant Run，点击Proceed Without Instant Run。此外，你需要在设备中启用开发调试选项才能插入Android设备。
    使用Android Studio将Tensorflow添加到您的应用程序
        allprojects {
            repositories {
                jcenter()
            }
        }
        dependencies {
            compile 'org.tensorflow:tensorflow-android:+'
        }
        但是加上dependencies {
                compile 'org.tensorflow:tensorflow-android:+'
            }会报错，
         Gradle sync failed: Could not find method compile() for arguments [org.tensorflow:tensorflow-android:+] on object of type
          org.gradle.api.internal.artifacts.dsl.dependencies.DefaultDependencyHandler.					Consult IDE log for more details (Help | Show Log) 
    Error:Gradle sync failed: Cannot create directory /home/xiang/.android/build-cache/3.2.1
死活安装不了，as一直在运行，也不知道进度，只知道在不断的弹出各种报错，也没有停下来的意思....

    def bazel_location = 'usr/local/bin/bazel'修改路径
    
    Configuration ‘compile’ is obsolete and has been replaced with ‘implementation’ and ‘api’.
    It will be removed at the end of 2018. For more information see: http://d.android.com/r/tools/update-dependency-configurations.html
        有些compile要改成implementation
    The minSdk version should not be declared in the android manifest file. You can move the version from the manifest to the defaultConfig in the build.gradle file.
    The targetSdk version should not be declared in the android manifest file. You can move the version from the manifest to the defaultConfig in the build.gradle file.
        minSdk version和targetSdk version不应该在manifest中声明，移动到build.gradle中
        <uses-sdk android:targetSdkVersion="23" />


####官方安装
在网上看到所有的./configure选择N（XLA，CUDA，MPI，..）除了android相关选项
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/README.md
    下载好模型解压到//tensorflow/examples/android/assets
    curl -L https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -o /tmp/inception5h.zip 
    curl -L https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1.zip -o /tmp/mobile_multibox_v1.zip 
    curl -L https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip -o /tmp/stylize_v1.zip

    bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
       --crosstool_top=//external:android/crosstool \
       --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
       --cxxopt=-std=c++11 \
       --cpu=armeabi-v7a

    bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
    
    然后不知道怎么做；就跳到https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
    bazel build --cxxopt='--std=c++11' -c opt //tensorflow/examples/android:tensorflow_demo
    然后成功了，
    adb install -r bazel-bin/tensorflow/examples/android/tensorflow_demo.apk
    装上了，稀里糊涂装上了










https://blog.csdn.net/masa_fish/article/details/54097796





一、下载 TensorFlow 项目源码//tensorflow/examples/android
二、安装Bazel
https://blog.csdn.net/masa_fish/article/details/54096996

安装tensorflow
adb devices没有反应，换个usb接口就好啦



三、安装 SDK
wget https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz----https://www.androiddevtools.cn/
tar xvzf android-sdk_r24.4.1-linux.tgz -C ~/tensorflow

cd  ~/tensorflow/android-sdk-linux
sudo  tools/android  update  sdk --no-ui
    tools/android java not found
        见二



四、安装 NDK
wget https:------https://www.androiddevtools.cn/
unzip android-ndk-r12b-linux-x86_64.zip -d ~/tensorflow
