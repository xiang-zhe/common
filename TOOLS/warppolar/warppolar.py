import cv2
import numpy as np
img = cv2.imread("polar_remap_doc.png")
dst = cv2.warpPolar(img,(176, 1106), (234,208), 176, cv2.INTER_NEAREST + cv2.WARP_POLAR_LINEAR)
cv2.namedWindow('r',0)
cv2.resizeWindow('r', 640,480)
cv2.imshow('r', dst)
cv2.waitKey(0)
cv2.imshow('r', img)
cv2.waitKey(0)
