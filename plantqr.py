
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


plant = cv2.imread("plant2.png")
qr = cv2.imread("plant.jpg")
qr = cv2.resize(qr,(150, 150), interpolation = cv2.INTER_CUBIC)

#cv2.imshow("image", plant)

#grayI = cv2.cvtColor(plant, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", grayI)

#thresh = cv2.adaptiveThreshold(grayI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)



h, w, d = qr.shape

# Fourth Quadrant of image
#C = plant[int(h/2):int(h),int(w/2):int(w)]
#cv2.imshow('img', C)


plant[750:750+h, 1300:1300+w] = qr
cv2.imshow("qr", plant)

cv2.imwrite("plantqr.jpg", plant)

cv2.waitKey(0)