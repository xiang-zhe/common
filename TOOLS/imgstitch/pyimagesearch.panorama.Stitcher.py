from pyimagesearch.panorama import Stitcher
import imutils
import cv2
imageA = cv2.imread('./images/bryce_left_01.png')
imageB = cv2.imread('./images/bryce_right_01.png')
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)

