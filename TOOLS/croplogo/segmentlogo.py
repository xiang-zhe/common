import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

background = './frame.png'
target = './y.jpg'
back = cv2.imread(background)
img = cv2.imread(target)
img = cv2.resize(img,None,fx=0.2,fy=0.2)
rows,cols,channel = img.shape
center = [back.shape[0]/2, back.shape[1]/2]
print(img.shape)
#print(img)
#B,G,R = img
for m in range(rows):
	for n in range(cols):
		#print(img[m,n].all, type(img[m,n]))np.array([255,255,255])
		if (img[m,n] == 255).all():
			#print(m, n)
			continue
		else:
			back[int(center[0]-rows/2+m), int(center[1]-cols/2+n)] = img[m,n]
cv2.imwrite('framegen.png', back)

