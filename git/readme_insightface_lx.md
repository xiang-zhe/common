demo代码修改：
face_model.py
  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      print(404, 'no face found')
      return None
    bbox, points = ret
    #print(bbox.shape, points.shape)
    aligneds = []
    boxes = []
    '''
    if bbox.shape[0]==0:
      return None
      '''
    for box, point in zip(bbox,points):
      #bbox = bbox[0,0:4]
      point = point.reshape((2,5)).T
      #print(bbox)
      #print(points)
      nimg = face_preprocess.preprocess(face_img, box[:4], point, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      aligneds.append(aligned)
      boxes.append(box)
      ##print(aligned.shape,type(aligned))
    return aligneds, boxes
    
    
test.py
def comparer(reg,f,threshold=1.24):
    name = 'unknow'
    mindist = 10
    for key,value in reg.items():   
        dist = np.sum(np.square(f-value))
        if dist < threshold and dist < mindist:
            mindist = dist      
            name = key           
    return name,mindist
def getf(img):
    if model.get_input(img) == None:
        return None
    else:
        imgs,boxes = model.get_input(img)
        info = []
        for img,box in zip(imgs,boxes):
            try:
                #print(box)
                #print(img)
                f = model.get_feature(img)
                gender, age = model.get_ga(img)
                info.append([box,f,gender,age])
            except:
                print(-10010)
        return info

model = face_model.FaceModel(args)

reg = defaultdict()
for i in glob.glob('register/*.*'):
    name = i.split('/')[-1].split('.')[0]
    img = cv2.imread(i)
    aligneds, boxes = model.get_input(img)
    f = model.get_feature(aligneds[0])
    reg[name] = f

capture = cv2.VideoCapture(0)
if capture.isOpened():
    while 1:
        ret, frame = capture.read()
        if ret is True:
            img = frame
            info = getf(img)
            if info != None:
                
                for box,f,gender,age in info:
                    name,mindist = comparer(reg,f,args.threshold)
                    cv2.rectangle(frame,(int(box[0]),int(box[1])-35),(int(box[2]),int(box[3])),(255,255,255))
                    cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,255,255))
                    cv2.putText(frame,name+' - '+str(age)+' - ' +str(gender),(int(box[0])+6,int(box[1])-6),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),1)
            cv2.imshow('0',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
                




关于insightface data的下载：
使用github中wiki的VGG2数据集，https://pan.baidu.com/s/1c3KeLzy，其中图片已经切好为112*112的尺寸。


关于insightface调试过程：
CUDA_VISIBLE_DEVICES='1,2,3' nohup python -u train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 --data-dir /media/ubuntu/eed112d9-6e0c-436c-8c85-7ffba7730f0e/insightfacedata/vgg2/train_crop112/faces_vgg_112x112 --prefix ~/facedata/insightfacedata/models_embedding --per-batch-size 128 --logs ~/facedata/insightfacedata/logs >~/facedata/insightfacedata/logs/nohup_output20180817.out 2>&1
报错：out of memory ； mxnet.base.MXNetError: [21:30:31] src/storage/./pooled_storage_manager.h:108: cudaMalloc failed: out of memory
解决：下调per-batch-size 64，

CUDA_VISIBLE_DEVICES='1,2,3' nohup python -u train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 \
--data-dir /media/ubuntu/eed112d9-6e0c-436c-8c85-7ffba7730f0e/insightfacedata/vgg2/train_crop112/faces_vgg_112x112 \
--prefix ~/facedata/insightfacedata/models_embedding --per-batch-size 64 \
--logs ~/facedata/insightfacedata/logs >~/facedata/insightfacedata/logs/nohup_output20180817.out 2>&1
per-batch-size 64，当使用4*1080ti时基本不会报错，使用3*1080ti偶尔会报错 ，mxnet.base.MXNetError: [21:30:31] src/storage/./pooled_storage_manager.h:108: cudaMalloc failed: out of memory

数据集生成.rec，.idx

    1，src/data/glint2lst.py /xxx/glint testdata > /home/xxx/glint_test.lst
    2，src/eval/gen_glint.py --input /home/xxx/glint_test.lst --output my_result.bin {...other param}

    运行python glint2lst.py /data/glint_data msra,celebrity > glint.lst

    我利用mxnet\trunk\tools\im2rec.py生成lst文件
    再利用这lst文件，通过insightface\trunk\src\data\face2rec2.py 生成rec文件
    是否要修改这些地方，insightface\trunk\src\train_softmax.py读取数据。
    1、insightface\trunk\src\common\face_preprocess.py
    def parse_lst_line(line):
        vec = line.strip().split("\t")
        assert len(vec)>=3
        aligned = int(vec[0])
        image_path = vec[1]
        label = int(vec[2])
        bbox = None
        landmark = None
    。。。
    修改为：
    def parse_lst_line(line):
        vec = line.strip().split("\t")
        assert len(vec)>=3
        aligned = True#int(vec[0])
        image_path = vec[2]
        label = int(float(vec[1]))
        bbox = None
        landmark = None
    2、insightface\trunk\src\data\face2rec2.py
    def read_list(path_in):
        path_ = path_in[:-4] #获取文件所在目录
        with open(path_in) as fin:
        identities = []
        last = [-1, -1]
        id = 1
        while True:
        line = fin.readline()
        if not line:
        break
        item = edict()
        item.flag = 0
        item.image_path, label, item.bbox, item.landmark, item.aligned = face_preprocess.parse_lst_line(line)
        item.image_path = os.path.join(path, item.image_path) #图片路径
        if not os.path.exists( item.image_path):
        continue
        if not item.aligned and item.landmark is None:
        #print('ignore line', line)
        continue
        item.id = _id
        item.label = [label, item.aligned] => item.label = label #label
    是的 image_path和label换一下位置

        https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py

    作者：https://github.com/deepinsight/insightface/issues/256
    Generate lst file by calling src/data/glint2lst.py. For example:
        python glint2lst.py /data/glint_data msra,celebrity > glint.lst
    Call face2rec2.py to generate .rec file.：


    使用glint2lst生成.lst时，缺少lmk文件，
        如果使用官方的数据集自带lmk文件。
    summary: 
    使用im2rec.py生成.lst文件:
        python src/data/im2rec.py glint_child ~/LX/images_crop112/ --list --recursive
        需要指定 --list， 即生成.lst，否则使用.lst生成.rec和.idx
        需要指定 --recursive，即递归子文件夹，否则只读取指定文件夹中的图片作为0类
    使用im2rec.py生成.rec和.idx文件：
        python src/data/im2rec.py glint_child ~/LX/images_crop112/

    预训练模型：
        CUDA_VISIBLE_DEVICES='1,2,3' nohup python -u src/train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 \
        --data-dir /home/ubuntu/facenet_gitlab/PythonDev/project/insight_face/insightface/datasets  --prefix ~/facedata/insightfacedata/models_embedding \
        --pretrained /home/ubuntu/facenet_gitlab/PythonDev/project/insight_face/insightface/models/,0 --per-batch-size 64 \
        --logs ~/facedata/insightfacedata/logs >~/facedata/insightfacedata/logs/nohup_output20180901.out 2>&1 &
        .rec和.idx文件必须是train.rec和train.idx 
        --pretrain 参数必须是 --pretrained /home/ubuntu/facenet_gitlab/PythonDev/project/insight_face/insightface/models/,0，调用模型也是这个格式
        若--pretrained /home/ubuntu/facenet_gitlab/PythonDev/project/insight_face/insightface/models/model,0 则指定为model--0000.params

        pretrained最好加上ckpt=2，否则模型没有达到一定阈值，不会保存下来
        CUDA_VISIBLE_DEVICES='1,2,3' nohup python -u src/train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 \
        --data-dir /home/ubuntu/facenet_gitlab/PythonDev/project/insight_face/insightface/datasets  --prefix ~/facedata/insightfacedata/models_embedding \
        --pretrained /home/ubuntu/facenet_gitlab/PythonDev/project/insight_face/insightface/models/,0  --ckpt 2 --per-batch-size 64 \
        --logs ~/facedata/insightfacedata/logs >~/facedata/insightfacedata/logs/nohup_output20180903.out 2>&1 &


TVM:
  https://github.com/deepinsight/insightface/issues/475
  data_shape = (1,3,112,112)
  target = tvm.target.create("llvm -mcpu=haswell") ----GPU model#----> target = tvm.target.cuda("llvm device=0")
  module.run(data=input_data)
  f1 = module.get_output(0).asnumpy()
  
