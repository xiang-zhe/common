import numpy as np
import cv2 
from PIL import Image, ImageDraw, ImageFont

#至少10个点匹配
MIN_MATCH_COUNT = 10
# 特征点提取方法，内置很多种
featureMethod = {
    "SIFT": cv2.xfeatures2d.SIFT_create(),
    "SURF": cv2.xfeatures2d.SURF_create(500), 
    "ORB": cv2.ORB_create()
    }


def get_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def match2frames(image1, image2, ):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    size1 = img1.shape
    size2 = img2.shape

    #img1 = cv2.resize(img1, (int(size1[1]*0.3), int(size1[0]*0.3)), cv2.INTER_LINEAR)
    #img2 = cv2.resize(img2, (int(size2[1]*0.3), int(size2[0]*0.3)), cv2.INTER_LINEAR)

    sift = featureMethod["SIFT"]

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))
    if len(good) <= MIN_MATCH_COUNT:
        return -1 # 完全不匹配 
    else:
        dis = []
        distance_sum = 0 # 特征点2d物理坐标偏移总和
        for m in good:
            dis.append(get_distance(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))
            #distance_sum += get_distance(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt)
        
        #distance = distance_sum / len(good) #单个特征点2D物理位置平均偏移量
        distance = np.argmax(np.bincount(np.array(dis, dtype = int)))
        return distance 


def main(minDev=5):
    cap = cv2.VideoCapture(0)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    first_frame = True
    index = 0
    dis = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if first_frame:
                pre_frame = frame
                first_frame = False
                continue
            index += 1
            if index == 24:
                dis = match2frames(pre_frame, frame)
                print("===>", dis)
                if 0 < dis < minDev: # catch the current nondeviation frame
                    pre_frame = frame
                index = 0
            size = frame.shape
            if size[1] > 720: # resize
                frame = cv2.resize(frame, (int(size[1]*0.5), int(size[0]*0.5)), cv2.INTER_LINEAR)
            #text_frame = putText(frame, dis)
            cv2.putText(frame, str(dis), (123,456), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 3)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
