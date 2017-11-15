
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

width = 864
#width = 1812
#height = 1007
height = 1152

#plant = cv2.imread("plantqr.jpg")
plant = cv2.imread("newplant1.jpg")
plant = cv2.resize(plant,(width, height), interpolation = cv2.INTER_CUBIC)



grayI = cv2.cvtColor(plant, cv2.COLOR_BGR2GRAY)
gray = np.float32(grayI)

h, w, d = plant.shape

thresh = cv2.adaptiveThreshold(grayI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)

shi = cv2.goodFeaturesToTrack(thresh, maxCorners = 50, qualityLevel = 0.7, minDistance = 10)

corners = np.int0(shi)

# Shi tomasi dots
'''
for i in corners:
    x,y = i.ravel()
    cv2.circle(plant,(x,y),5,255,-1)


plt.imshow(plant),plt.show()
'''
cv2.imshow("tg", thresh)

#use this for help later
#https://www.packtpub.com/mapt/book/application_development/9781785283932/5/ch05lvl1sec49/good-features-to-track





#https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
gray = cv2.bilateralFilter(grayI, 11, 17, 17)
edged = cv2.Canny(grayI, 30, 200)

cv2.imshow("edge", edged)

#(contours,_) = cv2.findContours(thresh, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

(_,contoursEdge,_) = cv2.findContours(edged.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

#D = cv2.drawContours(thresh, contours, contourIdx = -1, color = (0,0,255), thickness = 5)


cnts = sorted(contoursEdge, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
 
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

		
#Top left, Bottom left
#Top right, Bottom right
print (screenCnt)
print ("\n")


topL = screenCnt[0]
print (topL[0][1])

botL = screenCnt[1]
print (botL[0][1])

topR = screenCnt[2]
print (topR[0][0])

botR = screenCnt[3]
print (botR[0][0])

print ("\n")
y1 = screenCnt[0][0][1]
y2 = screenCnt[2][0][1]

x1 = screenCnt[0][0][0]
x2 = screenCnt[2][0][0]

print(y1)
print(y2)
print(x1)
print(x2)


cropped = plant[y1:y2, x1:x2] 
cv2.imshow("croppedImage", cropped)
#cv2.imwrite("qrextracted2.jpg", cropped)

cv2.drawContours(plant, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Game Boy Screen", plant)


cv2.waitKey(0)