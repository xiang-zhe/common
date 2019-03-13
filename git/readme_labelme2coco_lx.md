1, labelme
    标签应该使用--父类_子类_num方式， 如dog_golden_1, num表示对象到到个数

2, 
    2.1, labelme2coco.py
    能用，但是生成到json，不能使用cocoapi中visualization.py查看，key "id" error anns[ann["id']] = ann
    2.2, https://github.com/wucng/TensorExpand/blob/master/TensorExpand/Object%20detection/Data_interface/MSCOCO/labelme%20data/labelme2COCO.py
    labelme2coco_.py
        img = utils.img_b64_to_array(data['imageData'])
        ->img = utils.image.img_b64_to_arr(data['imageData'])
        
3， 
    验证json格式，https://github.com/wucng/TensorExpand/blob/master/TensorExpand/Object%20detection/Data_interface/MSCOCO/labelme%20data/visualization.py
    visualization.py
    
4， 数据就可以用于训练类，参见dataset_catalog.py
coco————coco_train2014
    |             |————1.jpg
    |             |————2.jpg
    |             |...
    ————annotations
                  |————coco_stuff_train.json
    ————coco_val2014
    |             |————1.jpg
    |             |————2.jpg
    |             |...
    ————annotations
                  |————coco_stuff_val.json
