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