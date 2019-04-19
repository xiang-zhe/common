import numpy as np
import cv2 
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

class stitcherAPI(object):
    def __init__(self, imgLeft=None, imgRight=None, featureMethod="sift", matchMethod='flann', goodnum=50, px=60, alpha_ =0.5, debug_ =0, SHOW = 0):
        self.imgLeft = imgLeft 
        self.imgRight = imgRight
        
        self.featureMethod = featureMethod 
        self.matchMethod = matchMethod
        self.goodnum = goodnum
    
        self.px = px   
        self.alpha_ = alpha_ 
        
        self.debug_  = debug_ 
        self.SHOW = SHOW
 
    def transformImg(self,):
        ######## preprocess
        global w1
        h1,w1,p1 = self.imgLeft.shape
        h2,w2,p2 = self.imgRight.shape
        h = np.maximum(h1,h2)
        _imgLeft = self.imgLeft.copy()
        _imgRight = self.imgRight.copy()
        _imgLeft[:int(0.15*h1), :, :] = 0
        _imgLeft[:, :int(0.5*w1), :] = 0
        _imgLeft[int(0.85*h1):, :, :] = 0
        #_imgLeft[:, int(0.85*w1):, :] = 0
        _imgRight[:int(0.15*h2), :, :] = 0
        #_imgRight[:, :int(0.15*w2), :] = 0
        _imgRight[int(0.85*h2):, :, :] = 0    
        _imgRight[:, int(0.5*w2):, :] = 0
        imgLeftgray = cv2.cvtColor(_imgLeft, cv2.COLOR_BGR2GRAY)
        imgRightgray = cv2.cvtColor(_imgRight, cv2.COLOR_BGR2GRAY)
        #imgLeftgray = cv2.cvtColor(self.imgLeft, cv2.COLOR_BGR2GRAY)
        #imgRightgray = cv2.cvtColor(self.imgRight, cv2.COLOR_BGR2GRAY)
        if self.debug_ :
            print('--debug--')
            if not os.path.exists('debug'):
                os.mkdir('debug')
        #第一选择是SURF,第二选择才是ORB
        ######## extractFeatures       
        if self.featureMethod == "orb":
            #### ORB
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(_imgLeft,None) ## des was array.uint8
            kp2, des2 = orb.detectAndCompute(_imgRight,None)
        elif self.featureMethod == "sift" :
            #### sift
            nfeatures = 0 #the number of best features to retain
            sift = cv2.xfeatures2d.SIFT_create() 
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(imgLeftgray,None) ## des was array.unit8  
            kp2, des2 = sift.detectAndCompute(imgRightgray,None)
        elif self.featureMethod == "surf":
            #### surf
            hessianThreshold = 500 # value bigger features less, whose hessian is larger than hessianThreshold are retained by the detector, 300-500
            surf = cv2.xfeatures2d.SURF_create(float(hessianThreshold))
            kp1, des1 = surf.detectAndCompute(imgLeftgray,None) ## des was array.float32,(<<1)
            kp2, des2 = surf.detectAndCompute(imgRightgray,None)
        else:
            print('Not support {} !'.format(self.featureMethod))
            os._exit(1) 
        #cv2.error: OpenCV(3.4.4) /io/opencv_contrib/modules/xfeatures2d/src/sift.cpp:1207: error: (-213:The function/feature is not implemented) This algorithm is patented and is excluded in this configuration; Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'create'
        #  pip3 install opencv-contrib-python==3.4.2.17  -i https://pypi.tuna.tsinghua.edu.cn/simple
        # 使用特征提取过程得到的特征描述符（descriptor）数据类型有的是float类型的，比如说SurfDescriptorExtractor， SiftDescriptorExtractor，有的是uchar类型的，比如说有ORB，BriefDescriptorExtractor。 对应float类型的匹配方式有：FlannBasedMatcher，BruteForce<L2<float>>，BruteForce<SL2<float>>，BruteForce<L1<float>>。 对应uchar类型的匹配方式有：BruteForce<Hamming>，BruteForce<HammingLUT>。所以ORB和BRIEF特征描述子只能使用BruteForce匹配法。
        # but i changed des to astype(np.float32), flannmatcher also works in orb; but worse when surf and sift to unit8,  
        if self.debug_ :
            print('des1: ',type(des1), des1)
            print('kp1: ',type(kp1), len(kp1), kp1[0])
        
        good = []
        if self.matchMethod == 'flann':
            #if self.featureMethod == 'orb':
            #    print('Not support ORB for {} !'.format(self.matchMethod))
            #    os._exit(1)
            FLANN_INDEX_KDTREE = 0  #建立FLANN匹配器的参数
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  ##配置索引，密度树的数量为5
            search_params = dict(checks = 50)   #指定递归次数
            #FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
            flann = cv2.FlannBasedMatcher(index_params, search_params)  #建立匹配器
            matches = flann.knnMatch(des1.astype(np.float32),des2.astype(np.float32),k=2)  #得出匹配的关键点  match(), radiusMatch() ,knnMatch()
            matchesImg = cv2.drawMatchesKnn(imgLeftgray, kp1, imgRightgray, kp2, matches, flags=0, outImg=None)
            # ratio test as per Lowe's paper
            for m, n in matches:
                if m.distance < 0.7*n.distance: ##value smaller features less
                    good.append(m)
        elif self.matchMethod[:2] == 'bf':
            if self.featureMethod == 'orb':
                distanceMethod = cv2.NORM_HAMMING
            else:
                distanceMethod = cv2.NORM_L2        ## NORM_L1,L2 are better for sift and surf
            bf = cv2.BFMatcher(distanceMethod, crossCheck=True) 
            if self.matchMethod == 'bf-knn':
                matches = bf.knnMatch(des1, des2, k=1)   #assert k ==1
                ## matches[0]=> 'list' object has no attribute 'distance'  ##element of matches is still a list, like [object]
                ## #print(matches) #[[<DMatch 0x7f4bbb9e1990>], [], [], [<DMatch 0x7f4bbb9e19b0>],...] lots of empty list
                if self.debug_ :
                    pass
                matches = list(x[0] for x in matches if x != [])
            elif self.matchMethod == 'bf':
                matches = bf.match(des1,des2) ## bf.match return [<DMatch 0x7f4bbb9e1990>, <DMatch 0x7f4bbb9e1990>,..]
            else:
                print('Not support {} !'.format(self.matchMethod))
                os._exit(1)
            matches = sorted(matches, key = lambda x:x.distance)
            matchesImg = cv2.drawMatches(self.imgLeft,kp1,self.imgRight,kp2,matches, flags=0,outImg=None )
            good = matches   
        else:
            print('Not support {} !'.format(self.matchMethod))
            os._exit(1)
        good = good[:self.goodnum] if len(good) > self.goodnum else good[:]
        if len(good) < 4:
            print('Less matches pairs, {}  !'.format(len(good)))
            os._exit(1)
        goodMatchesImg =cv2.drawMatches(self.imgLeft,kp1,self.imgRight,kp2,good, flags=0,outImg=None )

        if self.debug_ :
            print('matches: ', len(matches))
            print('matches[0]: ', matches[0])
            print('good: ',len(good))  ##good[0].shape=> 'cv2.DMatch' object has no attribute 'shape'
            print('good[0]: ',good[0])
            #cv2.imshow('0', matchesImg)
            #cv2.waitKey(0)
            #cv2.imshow('0', goodMatchesImg)
            #cv2.waitKey(0)
            cv2.imwrite('debug/matchesImg.jpg', matchesImg)
            cv2.imwrite('debug/goodMatchesImg.jpg',goodMatchesImg)

        ######## share;    images transformation
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)    #查询图像的特征描述子索引
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)    #训练(模板)图像的特征描述子索引
        M, mask=cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)         #find the M for warpPerspective


        ## should find a better method to estimate the average moving distance
        movedis_mode = np.argmax(np.bincount(abs((src_pts[:, 0, 0]-dst_pts[:, 0, 0]).astype(np.int))))
        
        if self.debug_ :
            print('M: ', M)
            print('dst_pts:', dst_pts)
            print('src_pts:', src_pts)
            print('dst_pts[:][:][0]', dst_pts[:, 0, 0]) ## different between dst_pts[:][0][0] and dst_pts[:, 0, 0]
            print('dst_pts', dst_pts.shape)
            print('movedis_mode:',movedis_mode)        
            print('dis:', src_pts[:, 0, 0]-dst_pts[:, 0, 0])
            
        warpImg = cv2.warpPerspective(self.imgRight,M,(w2+movedis_mode,h))  ##warp to big image  ##(w2+movedis_mode,h) should be target.size() 
        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        srcImg = cv2.warpAffine(self.imgLeft,M1,(w2+movedis_mode, h)) ##warp to big image
        
        return srcImg, warpImg, goodMatchesImg, M, movedis_mode 
        
    def transformImg2(self, M, movedis_mode):
        warpImg = cv2.warpPerspective(self.imgRight,M,(w2+movedis_mode,h))
        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        srcImg = cv2.warpAffine(self.imgLeft,M1,(w2+movedis_mode, h))
        return srcImg, warpImg

    def edgeMerge(self, srcImg, warpImg, overlay=True):  
        #assert srcImg.shape == warpImg.shape
        rows, cols= srcImg.shape[:2]
        dst_target = np.zeros([rows, cols, 3], np.uint8)
        if self.px == -1:      
            left = 0
            right = 0
            for col in range(0, cols):
                if srcImg[:, col].any() and warpImg[:, col].any(): ##np.array([0,0,0]).any() = 0 else =1 
                    left = col  ##look up left edge
                    break
            for col in range(cols-1, 0, -1):
                if srcImg[:, col].any() and warpImg[:, col].any():
                    right = col  ##look up right edge
                    break
            for row in range(0, rows):     
                for col in range(0, cols):
                    if not srcImg[row, col].any():  ## replace use imageTransfrom when src is empty
                        dst_target[row, col] = warpImg[row, col]
                    elif not warpImg[row, col].any():  ## reverse
                        dst_target[row, col] = srcImg[row, col]
                    else:  ## merge the coincidence region
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        dst_target[row, col] = np.clip(srcImg[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)
        else:    
            #self.px =30
            #self.alpha_ = 0.5
            '''
            for col in range(cols-1, 0, -1):
                if srcImg[:, col].any():
                    w1 = col  ##look up right edge
                    break'''
            _w = w1-int(self.px/2)
            dst_target[:, :w1-self.px] = srcImg[:, :w1-self.px]
            dst_target[:, w1:] = warpImg[:, w1:]
            unit_alpha = (1-self.alpha_)/(self.px/2)       
            for col in range(w1-self.px,_w):
                alpha = unit_alpha*float(self.px/2-abs(col-_w))
                #print(alpha)
                if overlay:
                    dst_target[:, col] = np.clip(srcImg[:, col] * (1-alpha) + warpImg[:, col] * alpha, 0, 255) 
                else:
                    dst_target[:, col] = np.clip(srcImg[:, col] * (1-alpha) , 0, 255) 
            #print('----')        
            for col in range(_w,w1):
                alpha = unit_alpha*float(self.px/2-abs(col-_w))        
                #print(alpha)
                if overlay:   
                    dst_target[:, col] = np.clip(srcImg[:, col] * alpha + warpImg[:, col] * (1-alpha), 0, 255)
                else:
                    dst_target[:, col] = np.clip(warpImg[:, col] * (1-alpha), 0, 255)

        return dst_target
        
    def main(self):  
        dirName, fileName = os.path.split(os.path.abspath(__file__))
        #print(dirname, filename)
        #rootPath = os.path.realpath(__file__) 
        dataPath = os.path.join(dirName, '6')
        #GOOD_POINTS_LIMITED = 0.99
        #src = 'w1.png'; des = 'w2.png'
        #src = '1.png'; des = '2.png'
        #src = os.path.join(dataPath, 'w1.png'); des = os.path.join(dataPath, 'w2.png')
        src = os.path.join(dataPath, '_left.jpg'); des = os.path.join(dataPath, '_right.jpg')
        self.imgLeft = cv2.imread(src,1)# 
        self.imgRight = cv2.imread(des,1)# 

        #dst = cv2.add(srcImg,warpImg)
        #dst_no = np.copy(dst)
        srcImg, warpImg, goodMatchesImg, M, movedis_mode = self.transformImg()

        dst_target = self.edgeMerge(srcImg, warpImg)
        #### other merge methods
        ##1 dst_target = np.maximum(srcImg,warpImg)  ##take the max

        ##2 dst_target = srcImg.copy()  ##take the src
        ## dst_target[:, w1:, :] = warpImg[:, w1:, :]

        if self.SHOW == 1:
            fig = plt.figure (tight_layout=True, figsize=(8, 18))
            gs = gridspec.GridSpec (6, 2)
            ax = fig.add_subplot (gs[0, 0])
            ax.imshow(self.imgLeft)
            ax = fig.add_subplot (gs[0, 1])
            ax.imshow(self.imgRight)
            ax = fig.add_subplot (gs[1, :])
            ax.imshow(matchesImg)
            ax = fig.add_subplot (gs[2, :])
            ax.imshow(warpImg)
            ax = fig.add_subplot (gs[3, :])
            ax.imshow(srcImg)
            ax = fig.add_subplot (gs[4, :])
            ax.imshow(dst_no)
            ax = fig.add_subplot (gs[5, :])
            ax.imshow(dst_target)
            #ax.set_xlabel ('The smooth method is SO FAST !!!!')
            plt.show()
            fig.savefig(os.path.join(dataPath, 'show.png'))
        else:
            cv2.imwrite(os.path.join(dataPath, 'goodmatches.jpg'),goodMatchesImg)
            cv2.imwrite(os.path.join(dataPath, 'warpImg.jpg'),warpImg)
            cv2.imwrite(os.path.join(dataPath, 'srcImg.jpg'),srcImg)
            cv2.imwrite(os.path.join(dataPath, 'tar_'+str(self.goodnum)+'_'+str(self.alpha_) +'_'+ str(self.px)+'.jpg'),dst_target)
            
if __name__ == '__main__':
    s = stitcherAPI(object)
    s.main()
    
    
