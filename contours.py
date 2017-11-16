
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


plant = cv2.imread("plant2Copy.png")


grayImg = cv2.cvtColor(plant, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(grayImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)

#gray = np.float32(grayImg)
#gray = cv2.bilateralFilter(gray, 11, 17, 17)



edged = cv2.Canny(thresh, 30, 200)


(_,contoursEdge,_) = cv2.findContours(edged.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)


print(np.shape(contoursEdge))

# Bounding rectangle
# x,y,w,h = cv2.boundingRect(contoursEdge)
# cv2.rectangle(plant, (x,y), (x+w, y+h), (0,255,0), 2)

# Hull around contours
hull = cv2.convexHull(contoursEdge)
cv2.polylines(plant, pts=hull, isCLosed=True, color=(0,255,255))


img = cv2.drawContours(plant, contoursEdge, contourIdx=-1, color=(0,0,255), thickness=2)

cv2.imshow("imgEdge", edged)
cv2.imshow("img", img)

cv2.waitKey(0)