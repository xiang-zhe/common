1, labelme
  标签应该使用--父类_子类_num方式， 如dog_golden_1, num表示对象到到个数

2, 
  2.1, labelme2coco.py
  能用，但是生成到json，不能使用cocoapi中visualization.py查看，key "id" error anns[ann["id']] = ann
  里面没有生成annotation['id']字段，#这个id表示annotation的id，因为每一个图像有不止一个annotation，所以要对每一个annotation编号
  2.2, https://github.com/wucng/TensorExpand/blob/master/TensorExpand/Object%20detection/Data_interface/MSCOCO/labelme%20data/labelme2COCO.py
    labelme2coco_.py
      img = utils.img_b64_to_array(data['imageData'])
      ->img = utils.image.img_b64_to_arr(data['imageData']）
      ann["area"]中没有这个字段，但是mask_rcnn需要，https://github.com/wucng/TensorExpand/issues/2
      程序中添加 # 计算轮廓面积
      polygon = np.array([points], dtype=np.int32)  # 这里是多边形的顶点坐标
      #im = np.zeros(image.shape[:2], dtype="uint8")  # 获取图像的维度: (h,w)=iamge.shape[:2]
      im = np.zeros((4000,4000), dtype="uint8") #假定一个画布
      polygon_mask = cv2.fillPoly(im, polygon, 255)
      annotation['area'] = float(np.sum(np.greater(polygon_mask, 0)))
      或者'''不可用
      contour = PascalVOC2coco.change_format(annotation['segmentation'][0])
      annotation['area']=abs(cv2.contourArea(contour,True))'''
      或者'''不可用
      poly = Polygon(points)
      annotation['area'] = round(poly.area,6)
      '''
        
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
