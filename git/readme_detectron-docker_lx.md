docker: docker pull yidliu/detectron:maskrcnn


python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /tmp/detectron-visualizations \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo
    
python tools/test_net.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    NUM_GPUS 1

数据集位置写在dataset_catalog.py中

Q:
  1,  urllib2.HTTPError: HTTP Error 404: Forbidden
    A:https://github.com/facebookresearch/Detectron/issues/792
    修改网址s3-us-west-2.amazonaws.com/ to dl.fbaipublicfiles.com
  2，  AssertionError: Detectron only automatically caches URLs in the Detectron S3 bucket: https://s3-us-west-2.amazonaws.com/detectron
    A：https://github.com/facebookresearch/Detectron/issues/820
    修改_DETECTRON_S3_BASE_URL = 'https://dl.fbaipublicfiles.com/detectron' in line 40 in /detectron/lib/utils/io.py
  3，Exception encountered running PythonOp function: ValueError: could not broadcast input array from shape (4) into shape (0)
    A: The question was from the 'NUM_CLASS' in the xxx.yaml file
    
    
    

官网docker：
cd $DETECTRON/docker
docker build -t detectron:c2-cuda9-cudnn7 .

nvidia-docker run --rm -it detectron:c2-cuda9-cudnn7 python detectron/tests/test_batch_permutation_op.py

docker:
    rename /etc/apt/source.list.d /etc/apt/source.list.d.bak    
    apt-get update
    apt-get install vim
    
Q：
    1，Could not find a package configuration file provided by "gflags" with any of the following names:
    or The command '/bin/sh -c make ops' returned a non-zero code: 2
        A:https://github.com/facebookresearch/Detectron/issues/756
        修改dockerfile部分即可
        添加
        WORKDIR /detectron
        RUN git checkout d56e267efc92b65b8d899f1b89a7ed2bca3e5f44
        #
        RUN pip install -r /detectron/requirements.txt
        1.1， error: Your local changes to the following files would be overwritten by checkout:
        Please, commit your changes or stash them before you can switch branches. Aborting
            A:https://www.jianshu.com/p/f27b33343afa
            # Clone the Detectron repository
            RUN git clone https://github.com/facebookresearch/detectron /detectron
            #RUN cd /detectron && git checkout d56e267efc92b65b8d899f1b89a7ed2bca3e5f44
            RUN cd /detectron && git reset --hard && git pull && git checkout d56e267      
    2，Get:23 http://us-east-1.ec2.archive.ubuntu.com/ubuntu xenial/multiverse amd64 Packages [176 kB]
    0% [Working]
        A:卡住不动，换源也不行，，重装镜像也不行
        原来是/etc/apt/source.list.d的问题，删掉就好了
    3，yaml.constructor.ConstructorError: while constructing a Python instance expected a class, but found <class 'builtin_function_or_method'> in "<unicode string>", line 3, column 20: BBOX_XFORM_CLIP: !!python/object/apply:numpy.core ...
        A:I rolled the version back to 4.2 and it works fine:     pip install pyyaml==4.2b2
    4，File "/home/qifengliang/detectron/detectron/utils/vis.py", line 332, in vis_one_image
    e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    ValueError: not enough values to unpack (expected 3, got 2)
        A: contour, hier = cv2.findContours( e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
