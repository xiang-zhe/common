import cv2
import numpy as np

class CV2FUNC:
    def __init__(self,):
        pass
    
    @staticmethod
    def cv2show(im,w=1280,h=720):
        cv2.namedWindow("im", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("im", w, h)
        cv2.imshow("im", im)
        if cv2.waitKey(0)& 0xFF==27:
            cv2.destroyAllWindows()

    @staticmethod
    def cv2puttext(im,context="none", position=None, color=(255,255,255),font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
        if position is None:
            position  = (int(im.shape[1]*0.15),int(im.shape[0]*0.15))
        return cv2.putText(im, context, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

    @staticmethod
    def cv2line(im,line,color=(255,0,0),thickness=3):
        assert len(line) == 2
        return cv2.line(im, line[0], line[1], color, thickness)

    @staticmethod
    def cv2circle(im, circle):
        return cv2.circle(im, (circle[0],circle[1]), circle[2], (0,0,255), 3)

    @staticmethod
    def cv2ellipse(im, ellipse):
        return cv2.ellipse(im, ellipse, (0,255,255), 3)

    @staticmethod
    def cv2rectangle(im, rect):
        if len(rect) == 2:
            return cv2.rectangle(im, rect[0], rect[1], (0,255,255), 3)
        elif len(rect) == 4:
            return cv2.rectangle(im, (rect[0],rect[1]), (rect[2],rect[3]), (0,255,0), 3)
        else:
            raise NotImplementedError(f"length of rect is expecting 2 or 4, but got {len(rect)} for parameters for cv2rectangle")

    @staticmethod
    def cv2polygon(im, polygon):
        return cv2.polylines(im, np.int32([polygon]), True, (0,255,0), 3)

    @staticmethod
    def cv2overlay(im, overlay, alpha=0.5):
        im_overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2B)
        alpha_overlay = cv2.cvtColor(np.zeros(im_overlay.shape, np.uint8), cv2.COLOR_GRAY2B)
        alpha_overlay[:, :, 0] = alpha*255
        alpha_overlay[:, :, 1] = alpha*255
        alpha_overlay[:, :, 2] = alpha*255
        im_overlay = im_overlay + (alpha_overlay*overlay.astype(np.float32)).astype(np.uint8)
        im_overlay = im_overlay.astype(np.uint8)
        im[:, :, 0] = im_overlay[:, :, 0]
        im[:, :, 1] = im_overlay[:, :, 1]
        im[:, :, 2] = im_overlay[:, :, 2]
        return im

    @staticmethod
    def cv2save(im,path):
        cv2.imwrite(path,im)
    @staticmethod
    def cv2gray(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    @staticmethod
    def cv2blur(im, ksize=(5,5), bk=0.0):
        return cv2.blur(im,ksize)




  
