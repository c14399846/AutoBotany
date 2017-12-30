
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


#plant = cv2.imread("plant2Copy.png")

plant = cv2.imread("plant2CopyCropped.png")
plant2 = cv2.imread("plant2CopyCropped.png")
plant3 = cv2.imread("plantqr.jpg")

pea = cv2.imread("PEA_9.png")

plant = pea
plant2 = pea


#TESTING OUT SATURATION THRESHOLDS
s = cv2.cvtColor(plant, cv2.COLOR_BGR2HSV)
_, s_thresh = cv2.threshold(s, 85, 255, cv2.THRESH_BINARY)

s_thresh = cv2.cvtColor(s_thresh, cv2.COLOR_HSV2BGR)
s_thresh = cv2.cvtColor(s_thresh, cv2.COLOR_BGR2GRAY)


cv2.imshow('saturation', s)
cv2.imshow('saturation_thresh', s_thresh)

# Used this for some help
# http://www.linoroid.com/2016/12/detect-a-green-color-object-with-opencv/
# Simple example stuff (the lower / higher ranges)
# 29 Dec 19:21
#lower_green = (65,60,60)
#upper_green = (80,255,255)

# THIS IS ONE IS REALLY GOOD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 29 Dec 19:33
lower_green = (30,60,60)
upper_green = (80,255,255)

hsvrange = cv2.inRange(s, lower_green, upper_green)

cv2.imshow('hsvrange', hsvrange)

# 30 Dec 2017 17:45
plantLoc = cv2.bitwise_and(pea, pea, mask = hsvrange)

cv2.imshow('plantLoc', plantLoc)

'''
plant = cv2.imread("newplant2.jpg")


width = 864
height = 1152

plant = cv2.resize(plant,(width, height), interpolation = cv2.INTER_CUBIC)
'''

grayImg = cv2.cvtColor(plant, cv2.COLOR_BGR2GRAY)

#gray = np.float32(grayImg)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#grayImg = clahe.apply(grayImg)

# 30 Dec 2017 17:50
#plantTEST = cv2.bilateralFilter(plantLoc, 11, 17, 17) # WILL FIND THE GREEN PLANT, NEED TO CHANGE THE 'RANGE' TO LOWER THAN 10 (NUMBER OF PLANTS IN IMAGE)


plantTEST = cv2.bilateralFilter(plant2, 11, 17, 17)
plantLAB = cv2.cvtColor(plantTEST, cv2.COLOR_BGR2LAB)


# MOVED FROM CODE BELOW
l, a, b = cv2.split(plantLAB)
#cv2.imshow('l_channel', l)
#cv2.imshow('a_channel', a)
#cv2.imshow('b_channel', b)


#plant2 = clahe.apply(plant2)
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

limg = cv2.merge((cl,a,b))
#cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final2 = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

grayImg2 = cv2.cvtColor(final2, cv2.COLOR_BGR2GRAY)

cv2.imshow("final2", final2)



# MOVED FROM CODE BELOW THIS
# Colour Range
#lower2 = (40, 85, 50)
#higher2 = (150, 190, 205)

# 29 Dec
#lower2 = (0, 5, 0)
#higher2 = (15, 100, 90)

lower2 = (0, 5, 0)
higher2 = (45, 190, 120)



#converted = cv2.cvtColor(plant2, cv2.COLOR_BGR2HSV)
roiColour2 = cv2.inRange(final2, lower2, higher2) # THIS ONE IS BETTER I THINK
#roiColour = cv2.inRange(final, lower, higher)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

cv2.imshow('roi2', roiColour2)

roiColour2 = cv2.dilate(roiColour2, kernel, iterations = 2)
roiColour2 = cv2.erode(roiColour2, kernel, iterations = 2)

skin2 = cv2.bitwise_and(final2, final2, mask = roiColour2)
#skin = cv2.bitwise_and(final, final, mask = roiColour)

cv2.imshow('skin2', skin2)







grayImg = cv2.bilateralFilter(grayImg, 11, 17, 17)
grayImg2 = cv2.bilateralFilter(grayImg2, 11, 17, 17)

#cv2.imshow("grayImg", grayImg)
cv2.imshow("grayImg2", grayImg2)

thresh = cv2.adaptiveThreshold(grayImg2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)

cv2.imshow('thresh', thresh)


edged = cv2.Canny(thresh, 30, 200)


(_,contoursEdge,_) = cv2.findContours(edged.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
#c = contoursEdge[0]


# Area
contoursEdge = sorted(contoursEdge, key = cv2.contourArea, reverse = True)

# Length
'''
L = None

for i in range(10):
	
	peri = cv2.arcLength(contoursEdge[i], True)
	#approx = cv2.approxPolyDP(contoursEdge[1], 0.01 * peri, True)

	print (peri)
	
	L = peri

contoursEdge = sorted(L, key = cv2.contourArea, reverse = True)
'''
	



# Bounding rectangle
#x,y,w,h = cv2.boundingRect(c)
#cv2.rectangle(plant, (x,y), (x+w, y+h), (0,255,0), 2)

# Hull around contours
#hull = cv2.convexHull(c)
#cv2.polylines(plant, pts=hull, isClosed=True, color=(0,255,255))



font = cv2.FONT_HERSHEY_SIMPLEX
# Finds 10 largest contours in the image, form sorted contours
for i in range(10):
	
	place = i
	x,y,w,h = cv2.boundingRect(contoursEdge[i])
	cv2.rectangle(plant, (x,y), (x+w, y+h), (0,255,0), 2)
	cv2.putText(plant,str(place),(x, (y-10)), font, 1,(255,255,255),2,cv2.LINE_AA)
	
	
	
	#hull = cv2.convexHull(contoursEdge[i])
	#cv2.polylines(plant, pts=hull, isClosed=True, color=(0,255,255))
	img = cv2.drawContours(plant, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness=2)

	
#cv2.imshow("grayImg", grayImg)
cv2.imshow("imgEdge", edged)
cv2.imshow("img", img)




# colour hist
#https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/

print ("START COLOURS\n")


'''
# convert the image to grayscale and create a histogram
grayy = cv2.cvtColor(plant, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", grayy)
hist = cv2.calcHist([grayy], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
'''


'''
# grab the image channels, initialize the tuple of colors,
# the figure and the flattened feature vector
chans = cv2.split(plant)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
#plt.show() 


print ("START LOOPING\n")
# loop over the image channels
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)
 
	# plot the histogram
	plt.plot(hist, color = color)
	plt.xlim([0, 256])
 
plt.show()
print ("FINISH LOOPING\n")
'''

# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
#print "flattened feature vector size: %d" % (np.array(features).flatten().shape)




##########################
#COLOUR WORK
# https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
# USING LAB FOR CLAHE CONTRAST FIXING

plantLAB = cv2.cvtColor(plant2, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(plantLAB)
#cv2.imshow('l_channel', l)
#cv2.imshow('a_channel', a)
#cv2.imshow('b_channel', b)


#plant2 = clahe.apply(plant2)
cl = clahe.apply(l)
#cv2.imshow('CLAHE output', cl)

limg = cv2.merge((cl,a,b))
#cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)






# VERY GOOD RANGE IN GENERAL
lower = (40, 85, 50)
higher = (165, 190, 160)

#converted = cv2.cvtColor(plant2, cv2.COLOR_BGR2HSV)
roiColour = cv2.inRange(plant2, lower, higher) # THIS ONE IS BETTER I THINK
#roiColour = cv2.inRange(final, lower, higher)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

cv2.imshow('roi', roiColour)

roiColour = cv2.dilate(roiColour, kernel, iterations = 2)
roiColour = cv2.erode(roiColour, kernel, iterations = 2)

skin = cv2.bitwise_and(plant2, plant2, mask = roiColour)
#skin = cv2.bitwise_and(final, final, mask = roiColour)

cv2.imshow('skin', skin)

##############
# END COLOURWORK




# testing qr stuff again
# it's the most distinct square thing, because I inserted it as such

gray3 = cv2.cvtColor(plant3, cv2.COLOR_BGR2GRAY)

gray3 = cv2.bilateralFilter(gray3, 11, 17, 17)

thresh3 = cv2.adaptiveThreshold(gray3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)


#gray3 = cv2.bilateralFilter(gray3, 11, 17, 17)
edged3 = cv2.Canny(thresh3, 30, 200)


(_,contoursEdge2,_) = cv2.findContours(edged3.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

cnts = sorted(contoursEdge2, key = cv2.contourArea, reverse = True)[:10]
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



cv2.drawContours(plant3, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("QRCode", plant3)





#exit()

cv2.waitKey(0)