import numpy as np
import cv2

# Load an color image and convert to grayscale
#img = cv2.imread('images/rgb_colorado.jpg')
img = cv2.imread('GOES12452016029pJxqq0-A.jpg')
#img = cv2.imread('images/rgb_northwest.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = img
cv2.imshow('rgb',img)
"""
# Need to explore more thresholding
(T, thresh) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
# (T, thresh) = cv2.threshold(gray, 130, 255, cv2.THRESH_TOZERO)

# Initial erode to remove lines and small noise
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 1)

# Dilation and erosion to extract lines
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
kernel = np.ones((3,3),np.uint8)
dilation = cv2.erode(dilation,kernel,iterations = 1)
diff = dilation-erosion

# Convert to BGR and eliminate B and R
lines = (cv2.cvtColor(diff,cv2.COLOR_GRAY2RGB))
lines[:,:,0] = 0
lines[:,:,1] = 0

# Combine lines with initial image
final = cv2.add(lines,img)

cv2.imshow('rgb',img)
#cv2.imshow('grey',gray)
cv2.imshow('thesh',thresh)
cv2.imshow('morphilogical',erosion)
cv2.imshow('morphilogical2',dilation)
cv2.imshow('diff',cv2.cvtColor(diff,cv2.COLOR_GRAY2RGB))
cv2.imshow('final',final)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
