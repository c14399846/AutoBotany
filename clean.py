import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lower_green = (30,60,60) # Original, some patches, little 'Noise'
#lower_green = (30,60,50) # Decent detection, catches more 'Noise'
#lower_green = (30,60,40)  # Less pathches, more 'Noise' # Lower plant Colourspace (HSV)
upper_green = (80,255,255) 								# Upper plant Colourspace (HSV)




# Read in plant image
def readInPlant(imagePath):

	plant = cv2.imread(imagePath)

	return plant



# Convert BGR to GRAY
def convertBGRGray(image):

	grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	return grayImg



# Converts BGR to HSV
def convertBGRHSV(image):

	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	return HSV



# Converts HSV to BGR
def convertHSVBGR(image):

	BGR = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

	return BGR



# Threshold of image
def getThreshold(image, lowerTH, upperTH):

	_, thresh = cv2.threshold(image, lowerTH, upperTH, cv2.THRESH_BINARY)

	return thresh



# Returns colours that match the colourspace range
def getColourRange(image, lower, upper):
	
	range = cv2.inRange(image, lower, upper)
	
	return range



# Seperates colours in image that match the mask
def getPlantLocation(image, range):

	plantLocation = cv2.bitwise_and(image, image, mask = range)
	
	return plantLocation
	


# Applies bilateral filter to an image
def applyBilateralFilter(image,d,sigmaColor,sigmaSpace):
	
	bilateral = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
	
	return bilateral
	


# Converts BGR to LAB colourspace
def convertBGRLAB(image):

	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	
	return lab



# Splits lab into its individual channels [l, a, b]
def splitLAB(image):
	
	l, a, b = cv2.split(image)
	
	return [l, a, b]



# Applies CLAHE filter to an image
def applyCLAHE(image):

	cl = clahe.apply(image)
	
	return cl



# Merge Colour channels
def mergeColourspace(first, second, third):

	merged = cv2.merge((first, second, third))

	return merged



# Merge LAB colour channels
def mergeLAB(l, a, b):
	
	mergedLAB = cv2.merge((l, a, b))

	return mergedLAB



# Converts LAB image to BGR
def convertLABBGR(image):

	bgr = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

	return bgr

	

	
# Shows all processed images
# WORK IN PROGRESS
def showImages():
	return
	


# Apply Canny Edge
def applyCanny(image, lowerEdge, upperEdge):

	canny = cv2.Canny(image,lowerEdge, upperEdge)
	
	return canny
	

	
# Apply Morphological Processes
def applyMorph(image):

	largeMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
	smallMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

	image = cv2.dilate(image,largeMorph)
	#image = cv2.erode(image,largeMorph)
	#image = cv2.dilate(image,smallMorph)
	
	return image

	
	
# Merges 2 Edge images together to find better Contours
def mergeEdges(img1, img2, imgShape):	
	
	rows,cols,channels = imgShape
	roi = img1[0:rows, 0:cols ]
	
	img2gray = img2.copy()
	ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
	
	mask_inv = cv2.bitwise_not(mask)
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
	img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
	
	dst = cv2.add(img1_bg,img2_fg)
	img1[0:rows, 0:cols ] = dst
	
	return img1
	


# Gets Contours using edges derived from a mask image
def getContours(plant, edge):

	# Makes a copy of original plant image 
	#(contours draw on the image supplied, use temp image)
	baseImg = plant.copy()

	
	# Finds contours (Have to be closed edges)
	(_,contoursEdge,_) = cv2.findContours(edge, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)


	# Find largest contours by Area (Have to be closed contours)
	contoursEdge = sorted(contoursEdge, key = cv2.contourArea, reverse = True)

	
	# Font for drawing text on image
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	
	# Finds largest contours in the image, from sorted contours
	
	##################################################################
	# NEED TO ADD THRESHOLD MIN SIZE FOR CONTOURS 
	# (if a plant hasn't sprouted yet, adds random contours)
	##################################################################
	
	for i in range(numberPlants):
		
		# Draw rectangles, with order of Contour size
		
		place = i
		x,y,w,h = cv2.boundingRect(contoursEdge[i])
		#cv2.rectangle(baseImg, (x,y), (x+w, y+h), (0,255,0), 2)
		#cv2.putText(baseImg,str(place),(x, (y-10)), font, 1,(255,255,255),2,cv2.LINE_AA)
		
		# Crops plant out of image, for later usage
		cropped = baseImg[y-5:y+h+5, x-5:x+w+5]
		cv2.imshow("cropped", cropped)

		#cv2.waitKey(0)
		
		# Other kind of Contouring
		#epsilon = 0.01*cv2.arcLength(contoursEdge[i],True)
		#approx = cv2.approxPolyDP(contoursEdge[i],epsilon,True)
		
		hull = cv2.convexHull(contoursEdge[i])
		cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
		img = cv2.drawContours(baseImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)

	return baseImg
	
	

# The full image process pipeline
def process(plantOrig):

	processedImages = [] # Array of processed images
	count = 0 			 # Integer that holds number of images processed + saved
	
	
	
	
	#################################################################
	# TEST CODE #
	
	#lower_green = (30,60,60)
	#upper_green = (80,255,255)
	
	kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
	kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)

	sharpOrig = cv2.filter2D(plantOrig, ddepth = -1, kernel = kernelSharp)
	cv2.imshow("sharpOrig", sharpOrig)
	
	# Has interesting seperation of Colours,
	# Need to get good ranges for it
	YUV = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2YUV)
	cv2.imshow("YUV", YUV)
	
	sharpenYUV = cv2.filter2D(YUV, ddepth = -1, kernel = kernelSharp)
	cv2.imshow("sharpenYUV", sharpenYUV)
	
	LAB = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2LAB)
	cv2.imshow("LAB", LAB)
	
	sharpenLAB = cv2.filter2D(LAB, ddepth = -1, kernel = kernelSharp)
	cv2.imshow("sharpenLAB", sharpenLAB)
	
	# END TEST CODE #
	#################################################################
	
	
	

	# Converts image to HSV colourspace
	# Gets colours in a certain range
	hsv = convertBGRHSV(plantOrig)
	hsvrange = getColourRange(hsv, lower_green, upper_green)
	if(addhsvrange):
		processedImages.append([])
		processedImages[count].append(hsvrange)
		processedImages[count].append("hsvrange")
		count += 1
	cv2.imshow("hsv", hsv)
	#cv2.imshow("hsvrange", hsvrange)

	sharpenHSV = cv2.filter2D(hsv, ddepth = -1, kernel = kernelSharp)
	cv2.imshow("sharpenHSV", sharpenHSV)
	
	# Applies filters to blend colours
	# *Might* make plant extraction easier (for edges / contours)
	bilateral = applyBilateralFilter(plantOrig, 11, 17, 17)
	
	
	# NOT AS USEFUL ANYMORE,
	# MAYBE FOR OTHER PLANTS / SCENARIOS
	'''
	lab = convertBGRLAB(bilateral)
	if(addlab):
		processedImages.append([])
		processedImages[count].append(lab)
		processedImages[count].append("lab")
		count += 1
	#cv2.imshow("lab", lab)
	
	# Split LAB colour channels, apply CLAHE
	labChannels = splitLAB(lab)
	l = applyCLAHE(labChannels[0])
	a = labChannels[1]
	b = labChannels[2]
	
	
	# Merge colour channels, convert back to BGR
	labSmoothed = mergeLAB(l, a, b)
	labBGR = convertLABBGR(labSmoothed)
	if(addlabBGR):
		processedImages.append([])
		processedImages[count].append(labBGR)
		processedImages[count].append("labBGR")
		count += 1
	#cv2.imshow("labBGR", labBGR)
	'''
	
	##########################################################
	#		TEST CODE CHANGED FROM 'labBGR' TO 'bilateral'	 #
	##########################################################
	
	# Convert Filtered image to HSV, get colour range for mask
	filtered = convertBGRHSV(bilateral)
	filteredRange = getColourRange(filtered, lower_green, upper_green)
	if(addfilteredRange):
		processedImages.append([])
		processedImages[count].append(filteredRange)
		processedImages[count].append("filteredRange")
		count += 1
	#cv2.imshow("addfilteredRange", addfilteredRange)
	
	
	
	# Finds Plant Pixels matching the Mask
	origImgLoc = getPlantLocation(plantOrig, hsvrange)
	if(addorigImgLoc):
		processedImages.append([])
		processedImages[count].append(origImgLoc)
		processedImages[count].append("origImgLoc")
		count += 1
	cv2.imshow("origImgLoc", origImgLoc)
	
	
	
	# Finds Plant Pixels matching the Filtered Mask
	filteredImgLoc = getPlantLocation(plantOrig, filteredRange)
	if(addfilteredImgLoc):
		processedImages.append([])
		processedImages[count].append(filteredImgLoc)
		processedImages[count].append("filteredImgLoc")
		count += 1
	#cv2.imshow("filteredImgLoc", filteredImgLoc)
	
	
	
	# It has an interesting result, might look at later
	#BGRImgLoc = convertHSVBGR(origImgLoc)
	
	
	
	# NOT A GOOD PLACE TO MORPH,
	# STILL LOOKING INTO IT
	'''
	morph1 = applyMorph(origImgLoc)
	morph2 = applyMorph(filteredImgLoc)
	
	cv2.imshow("orig", origImgLoc)
	cv2.imshow("morph1", morph1)
	cv2.imshow("filt", filteredImgLoc)
	cv2.imshow("morph2", morph2)
	'''
	
	
	
	# SHI-TOMASI
	#
	# Might want to make the points track per plant, rather than the whole image.
	# The whole image parsing could possibly skip some of the plants in the image.
	#
	#
	img = origImgLoc.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
	corners = np.int0(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(img,(x,y),3,255,-1)

	#plt.imshow(img),plt.show()
	
	
	
	
	
	
	# Gets Canny Edges of Plant Pixels
	edgeLoc = applyCanny(origImgLoc, 30, 200)
	if(addedgeLoc):
		processedImages.append([])
		processedImages[count].append(edgeLoc)
		processedImages[count].append("edgeLoc")
		count += 1
	#cv2.imshow("edgeLoc", edgeLoc)
	
	
	
	# Gets Canny Edges of Filtered Plant Pixels
	edgeFilteredLoc = applyCanny(filteredImgLoc, 30, 200)
	if(addedgeFilteredLoc):
		processedImages.append([])
		processedImages[count].append(edgeFilteredLoc)
		processedImages[count].append("edgeFilteredLoc")
		count += 1
	#cv2.imshow("edgeFilteredLoc", edgeFilteredLoc2)
	
	
	
	
	
	# Adds the 2 Canny Edges together, better Countour coverage achieved
	#
	# https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
	# Reference code
	edge1 = edgeLoc.copy()
	edge2 = edgeFilteredLoc.copy()
	shape = plantOrig.shape
	
	doubleEdge = mergeEdges(edge1, edge2, shape)
	if(adddoubleEdge):
		processedImages.append([])
		processedImages[count].append(doubleEdge)
		processedImages[count].append("doubleEdge")
		count += 1
	#cv2.imshow("doubleEdge", doubleEdge)
	
	
	# Finds Contours from Both Edges
	contourRes = getContours(plantOrig, doubleEdge)
	if(addcontourRes):
		processedImages.append([])
		processedImages[count].append(contourRes)
		processedImages[count].append("contourRes")
		count += 1
	#cv2.imshow("contourRes", contourRes)

	# Good in some cases,
	# Less good in larger plant sizes / shapes.
	# Made somewhat redundant by above Double-Edge mix.
	'''
	# Finds Contours from Edges
	contour = getContours(plantOrig, edgeLoc)
	if(addcontour):
		processedImages.append([])
		processedImages[count].append(contour)
		processedImages[count].append("contour")
		count += 1
	#cv2.imshow("con1", con1)
	
	
	# Finds Contours from Filtered Edges
	contourFiltered = getContours(plantOrig, edgeFilteredLoc)
	if(addcontourFiltered):
		processedImages.append([])
		processedImages[count].append(contourFiltered)
		processedImages[count].append("contourFiltered")
		count += 1
	#cv2.imshow("con2", con2) 
	'''
	
	if(showAll):
		#print (count)
		for i in range(count):
			cv2.imshow(processedImages[i][1], processedImages[i][0])
	cv2.waitKey(0)
	


# STARTS HERE
# OPENS FILE / SOMEHOW GETS FILE
# FROM STORAGE, OR FROM CAMERA*
# *(Need to add camera operations, maybe)


# Set bool to append / not append images to list
addhsvrange = False
addlab = False
addlabBGR = False
addfilteredRange = False
addorigImgLoc = False
addfilteredImgLoc = False

addedgeLoc = False
addedgeFilteredLoc = False
adddoubleEdge = True

#addcontour = True
#addcontourFiltered = True
addcontourRes = True


# Set bool to Show all images added to list
showAll = False

# Number of plants in image (Can be defined by user later on)
numberPlants = 2




file = easygui.fileopenbox()
plantImg = readInPlant(file)

height, width = plantImg.shape[:2]
resized = cv2.resize(plantImg,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)



cv2.imshow("plantImg", plantImg)







# Processing pipeline
process(plantImg)
	

	
gray = convertBGRGray(plantImg)
cv2.imshow("gray", gray)
	
'''
TEST CODE FOR DRAWING ON IMAGES
FROM LABS WEEK 2_2

'''
	
def draw(event,x,y,flags,param): 
	if event == cv2.EVENT_LBUTTONDOWN:
		X1 = x-100
		X2 = x+100
		
		Y1 = y-100
		Y2 = y+100
		
		print ('x1 ', X1)
		print ('x2 ', X2)
		print ('y1 ', Y1)
		print ('y2 ', Y2)

		if Y1 < 0:
			Y1 = 0
		if Y2 > size[0]:
			Y2 = size[0]
		
		if X1 < 0:
			X1 = 0
		if X2 > size[1]:
			X2 = size[1]
		
		I[Y1:Y2,X1:X2] = HSV[Y1:Y2,X1:X2]
		
		##This has issues doing the bottom and right sides correctly for osme reason*********
		cv2.rectangle(img = I, 
			pt1 = (X1,Y1), 
			pt2 = (X2,Y2), 
			color = (255,0,255), 
			thickness = 5)
		
		cv2.imshow("image", I)
	
	
	
	
#cv2.setMouseCallback("image", draw)






cv2.waitKey(0)
cv2.destroyAllWindows()

'''
TEST CODE FOR DRAWING ON IMAGES
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
'''

drawHSV = plantImg.copy()

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(drawHSV,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(drawHSV,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(drawHSV,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(drawHSV,(x,y),5,(0,0,255),-1)

	
#img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('drawHSV')
cv2.setMouseCallback('drawHSV',draw_circle)

while(1):
    cv2.imshow('drawHSV',drawHSV)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
	


cv2.destroyAllWindows()
cv2.waitKey(0)

	
	
'''
TEST CODE
# GETS THE ROI FROM DRAG AND DROP MOUSE CLICKS
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

'''	


# import the necessary packages
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)


 
# load the image, clone it, and setup the mouse callback function
image = plantImg #cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
 
# close all open windows
cv2.destroyAllWindows()		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
cv2.waitKey(0)