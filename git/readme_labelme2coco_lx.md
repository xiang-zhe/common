1, labelme

2, 
    2.1, labelme2coco.py
    能用，但是生成到json，不能使用cocoapi中visualization.py查看，key "id" error anns[ann["id']] = ann
    2.2, https://github.com/wucng/TensorExpand/blob/master/TensorExpand/Object%20detection/Data_interface/MSCOCO/labelme%20data/labelme2COCO.py
    labelme2coco_.py
        img = utils.img_b64_to_array(data['imageData'])
        ->img = utils.image.img_b64_to_arr(data['imageData'])
    
