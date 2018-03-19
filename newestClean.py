from __future__ import print_function
import pyzbar.pyzbar as pyzbar


import sys
import numpy as np
import cv2
#from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

import cv2.aruco as aruco

from DrawOver import DrawOver




clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lower_green = (30,60,60) # Original, some patches, little 'Noise'
#lower_green = (30,60,50) # Decent detection, catches more 'Noise'
#lower_green = (30,60,40)  # Less pathches, more 'Noise' # Lower plant Colourspace (HSV)
upper_green = (80,255,255)	# Upper plant Colourspace (HSV)
#upper_green = (80,255,190)	# Upper plant Colourspace (HSV)



# This is the lower and upper ranges of the background
lower_bg  = (20, 70, 200)
#upper_bg = (30, 200, 255)
upper_bg = (33, 150, 255) # This one rmeoved a lot of crap, the plant look decent too

lower_dirt = (20, 130, 25)
upper_dirt = (30, 255, 115)
#upper_dirt = (32, 255, 115) # These are good for removing dirt, and make a better contour too
#upper_dirt = (33, 255, 115)


lower_support = (19, 67, 70)
upper_support = (30, 252, 253)


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



# Merge Plant Locations
def mergeImages(image1, image2, wgtImg1, wgtImg2):
	
	mergedImages = cv2.addWeighted(image1, wgtImg1, image2, wgtImg2, 0)
	
	return mergedImages
	
	

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

	mergedColours = cv2.merge((first, second, third))

	return mergedColours



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
		#cv2.rectangle(baseImg, (x,y), (x+w, y+h), (255,0,0), 2)
		#cv2.putText(baseImg,str(place),(x, (y-10)), font, 1,(255,255,255),2,cv2.LINE_AA)		
		
		# Crops plant out of image, for later usage
		#cropped = baseImg[y-5:y+h+5, x-5:x+w+5]
		#cv2.imshow("cropped", cropped)
		#cv2.waitKey(0)
		
		# Other kind of Contouring
		#epsilon = 0.01*cv2.arcLength(contoursEdge[i],True)
		#approx = cv2.approxPolyDP(contoursEdge[i],epsilon,True)
		
		
		# Contours that wrap around the plant object
		#hull = cv2.convexHull(contoursEdge[i])
		#cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
		#img = cv2.drawContours(baseImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
		
		

		# Cropping is here (If it works.....)
		#baseImg = baseImg[y-5:y+h+5, x-5:x+w+5]
		baseImg = baseImg[y:y+h, x:x+w]
		
		#cv2.imshow("baseImg", baseImg)
		#cv2.waitKey(0)
		
	return baseImg




# Find the QR Code in the image
def decode(im) : 

	height, width = im.shape[:2]
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Find barcodes and QR codes  
	decodedObjects = pyzbar.decode((gray.tobytes(), width, height))
 
	# Print results
	for obj in decodedObjects:
		print('Type : ', obj.type)
		print('Data : ', obj.data,'\n')
	
	return decodedObjects


# Display the QR Code on the image passed into the function
# Display barcode and QR code location  
def display(im, decodedObjects):

	# Loop over all decoded objects
	for decodedObject in decodedObjects: 
		points = decodedObject.rect
		print(points)

		# If the points do not form a quad, find convex hull
		if len(points) > 4 : 
			hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
			hull = list(map(tuple, np.squeeze(hull)))
		else : 
			hull = points;

		# Number of points in the convex hull
		n = len(hull)


		X = hull[0]
		width = hull[2]
		Y = hull[1]
		height = hull[3]

		cv2.line(im, (X,Y), (X + width,Y), (255,0,0), 1) # top line
		cv2.line(im, (X,Y), (X, Y + height), (255,0,0), 1) # left line
		cv2.line(im, (X,Y + height), (X + width, Y + height), (255,0,0), 1) # bottom line
		cv2.line(im, (X + width, Y + height), (X + width, Y), (255,0,0), 1) # right line
	
	# Display results
	#cv2.imshow("Results", im);
	#cv2.waitKey(0);

	return im


# Display the QR Code on the image passed into the function
# Display barcode and QR code location  
#def qrcodeDimensions(im, decodedObjects):
def qrcodeDimensions(decodedObjects):

	X = 0
	Y = 0
	width = 0
	height = 0


	# Loop over all decoded objects
	for decodedObject in decodedObjects: 
		points = decodedObject.rect
		#print(points)

		X = points[0]
		width = points[2]
		Y = points[1]
		height = points[3]
		
		'''
		print(X)
		print(Y)
		print(width)
		print(height)

		cv2.line(im, (X,Y), (X + width,Y), (255,0,0), 1) # top line
		cv2.line(im, (X,Y), (X, Y + height), (255,0,0), 1) # left line
		cv2.line(im, (X,Y + height), (X + width, Y + height), (255,0,0), 1) # bottom line
		cv2.line(im, (X + width, Y + height), (X + width, Y), (255,0,0), 1) # right line
		'''
	
	# Display results
	#cv2.imshow("Results", im);
	#cv2.waitKey(0);

	return width, height


# THIS NEEDS SOME WORK
# Gets Contours using edges derived from a mask image
def getContoursWrap(plant, edge):

	# Makes a copy of original plant image 
	#(contours draw on the image supplied, use temp image)
	baseImg = plant.copy()

	
	# Finds contours (Have to be closed edges)
	(_,contoursEdge,_) = cv2.findContours(edge, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)


	# Find largest contours by Area (Have to be closed contours)
	contoursEdge = sorted(contoursEdge, key = cv2.contourArea, reverse = True)
	
	
	for i in range(numberPlants):
		
		# Contours that wrap around the plant object
		hull = cv2.convexHull(contoursEdge[i])
		cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
		img = cv2.drawContours(baseImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
		
	return baseImg
	

# The full image process pipeline
def process(plantOrig):

	processedImages = [] # Array of processed images
	count = 0 			 # Integer that holds number of images processed + saved
	
	
	
	# GRABCUT TEST CRAP
	# 12 mar 2018 11:41am
	'''
	img = plantOrig.copy()


	mask = np.zeros(img.shape[:2],np.uint8)

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	# Using rectangle for extraction

	rect = (685,414,1323,950)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]


	plantOrig = img
	'''
	
	# END GRABCUT TEST CRAP
	
	
	
	
	
	
	
	
	#################################################################
	# TEST CODE #
	#lower_green = (30,60,60)
	#upper_green = (80,255,255)
	
	kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
	kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)

	sharpOrig = cv2.filter2D(plantOrig, ddepth = -1, kernel = kernelSharp)
	#cv2.imshow("sharpOrig", sharpOrig)
	
	# Has interesting seperation of Colours,
	# Need to get good ranges for it
	YUV = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2YUV)
	#cv2.imwrite("./images/yuv.png", YUV)
	#cv2.imshow("YUV", YUV)
	
	sharpenYUV = cv2.filter2D(YUV, ddepth = -1, kernel = kernelSharp)
	#cv2.imwrite("./images/sharpenyuv.png", sharpenYUV)
	#cv2.imshow("sharpenYUV", sharpenYUV)
	
	
	LAB = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2LAB)
	#cv2.imwrite("./images/lab.png", LAB)
	#cv2.imshow("LAB", LAB)
	
	sharpenLAB = cv2.filter2D(LAB, ddepth = -1, kernel = kernelSharp)
	#cv2.imwrite("./images/sharpenlab.png", sharpenLAB)
	#cv2.imshow("sharpenLAB", sharpenLAB)
	
	
	YCB = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2YCrCb)
	#cv2.imwrite("./images/ycb.png", YCB)
	
	
	# whatever this is....
	#imgYCC = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2YCR_CB)
	#cv2.imwrite("./images/ycc.png", imgYCC)
	
	sharpenYCB = cv2.filter2D(YCB, ddepth = -1, kernel = kernelSharp)
	#cv2.imwrite("./images/sharpenycb.png", sharpenYCB)
	#cv2.imshow("sharpenYCB", sharpenYCB)
	
	
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
	#cv2.imwrite("./images/hsv.png", hsv)
	cv2.imshow("hsv", hsv)
	#cv2.imshow("hsvrange", hsvrange)

	
	
	
	# Finds Plant Pixels matching the Mask
	origImgLoc = getPlantLocation(plantOrig, hsvrange)
	if(addorigImgLoc):
		processedImages.append([])
		processedImages[count].append(origImgLoc)
		processedImages[count].append("origImgLoc")
		count += 1
	#cv2.imshow("origImgLoc", origImgLoc)
	
	
	
	sharpenHSV = cv2.filter2D(hsv, ddepth = -1, kernel = kernelSharp)
	#cv2.imshow("sharpenHSV", sharpenHSV)
	
	# Applies filters to blend colours
	# *Might* make plant extraction easier (for edges / contours)
	bilateral = applyBilateralFilter(plantOrig, 11, 17, 17)
	
	
	# Convert Filtered image to HSV, get colour range for mask
	filtered = convertBGRHSV(bilateral)
	filteredRange = getColourRange(filtered, lower_green, upper_green)
	if(addfilteredRange):
		processedImages.append([])
		processedImages[count].append(filteredRange)
		processedImages[count].append("filteredRange")
		count += 1
	#cv2.imshow("addfilteredRange", addfilteredRange)
	
	
	# Finds Plant Pixels matching the Filtered Mask
	filteredImgLoc = getPlantLocation(plantOrig, filteredRange)
	if(addfilteredImgLoc):
		processedImages.append([])
		processedImages[count].append(filteredImgLoc)
		processedImages[count].append("filteredImgLoc")
		count += 1
	#cv2.imshow("filteredImgLoc", filteredImgLoc)
	
	
	mergedPlantAreas = mergeImages(origImgLoc, filteredImgLoc, 0.5, 0.5)
	#cv2.imshow("mergedPlantAreas", mergedPlantAreas)
	
	
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
	
	
	
	
	
	
	
	
	####################################################################
	# TEST CODE
	
	
	edgeHSV = applyCanny(hsv, 30, 200)
	edgeSharpenHSV = applyCanny(sharpenHSV, 30, 200)
	
	edgeYUV = applyCanny(YUV, 30, 200)
	edgeSharpenYUV = applyCanny(sharpenYUV, 30, 200)
	
	edgeLAB = applyCanny(LAB, 30, 200)
	edgeSharpenLAB = applyCanny(sharpenLAB, 30, 200)
	
	edgeYCB = applyCanny(YCB, 30, 200)
	edgeSharpenYCB = applyCanny(sharpenYCB, 30, 200)
	
	#'''
	#cv2.imshow("HSV", hsv)
	#cv2.imshow("edgeHSV", edgeHSV)
	#cv2.imshow("sharpenHSV", sharpenHSV)
	#cv2.imshow("edgeSharpenHSV", edgeSharpenHSV)
	#'''
	
	#'''
	#cv2.imshow("YUV", YUV)
	#cv2.imshow("edgeYUV", edgeYUV)
	#cv2.imshow("sharpenYUV", sharpenYUV)
	#cv2.imshow("edgeSharpenYUV", edgeSharpenYUV)
	#'''
	
	#'''
	#cv2.imshow("LAB", LAB)
	#cv2.imshow("edgeLAB", edgeLAB)
	#cv2.imshow("sharpenLAB", sharpenLAB)
	#cv2.imshow("edgeSharpenLAB", edgeSharpenLAB)
	#'''
	
	#'''
	#cv2.imshow("YCB", YCB)
	#cv2.imshow("edgeYCB", edgeYCB)
	#cv2.imshow("sharpenYCB", sharpenYCB)
	#cv2.imshow("edgeSharpenYCB", edgeSharpenYCB)
	#'''
	

	yuvY = YUV[...,0]
	yuvU = YUV[...,1]
	yuvV = YUV[...,2]
	
	#cv2.imshow("yuvY", yuvY)
	#cv2.imshow("yuvU", yuvU)
	#cv2.imshow("yuvV", yuvV)
	
	#tmpGray = cv2.cvtColor(plantImg,cv2.COLOR_BGR2GRAY)
	#cv2.imshow("tmpGray", tmpGray)
	
	
	
	
	
	# YCB
	
	# Original YCB, in grayscale outputs
	ycbY = YCB[...,0]
	ycbC = YCB[...,1]
	ycbB = YCB[...,2]
	
	
	copyRED_YCB = YCB.copy()
	copyRED_YCB[:,:,0] = 0
	copyRED_YCB[:,:,1] = 0
	#cv2.imshow("copyRED_YCB", copyRED_YCB)
	
	copyGR_YCB = YCB.copy()
	copyGR_YCB[:,:,0] = 0
	copyGR_YCB[:,:,2] = 0
	#cv2.imshow("copyGR_YCB", copyGR_YCB)
	
	copyBL_YCB = YCB.copy()
	copyBL_YCB[:,:,1] = 0
	copyBL_YCB[:,:,2] = 0
	#cv2.imshow("copyBL_YCB", copyBL_YCB)
	
	# Trying to get colour representation of YCB colourspace
	# It failed.
	#ycbY = YCB[:,:,0]
	#ycbC = YCB[:,:,1]
	#ycbB = YCB[:,:,2]
	
	#cv2.imshow("ycbY", ycbY)
	#cv2.imshow("ycbC", ycbC)
	#cv2.imshow("ycbB", ycbB)
	
	
	cpYCB = YCB.copy()
	
	cpYCB[:,:,0] = 0
	#cpYCB[:,:,1] = 0
	cpYCB[:,:,2] = 0
	cpYCB_Y = cpYCB[...,0]
	cpYCB_C = cpYCB[...,1]
	cpYCB_B = cpYCB[...,2]
	
	ycb_image = cv2.merge([cpYCB_Y, cpYCB_C, cpYCB_B])
	outYCB = cv2.cvtColor(ycb_image, cv2.COLOR_BGR2YCrCb)
	cv2.imshow("outYCB", outYCB)
	
	
	
	
	# LAB
	
	labL = LAB[...,0]
	labA = LAB[...,1]
	labB = LAB[...,2]
	
	#cv2.imshow("labL", labL)
	#cv2.imshow("labA", labA)
	#cv2.imshow("labB", labB)
	
	
	cpLAB = LAB.copy()
	
	#cpLAB[:,:,0] = 0
	cpLAB[:,:,1] = 0
	cpLAB[:,:,2] = 0
	cpL = cpLAB[...,0]
	cpA = cpLAB[...,1]
	cpB = cpLAB[...,2]
	
	lab_image = cv2.merge([cpL, cpA, cpB])
	
	outLAB = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
	cv2.imshow("outLAB", outLAB)
	
	
	
	
	
	# HSV
	
	copyRED_HSV = hsv.copy()
	copyRED_HSV[:,:,0] = 0
	copyRED_HSV[:,:,1] = 0
	#cv2.imshow("copyRED_HSV", copyRED_YCB)
	
	copyGR_HSV = hsv.copy()
	copyGR_HSV[:,:,0] = 0
	copyGR_HSV[:,:,2] = 0
	#cv2.imshow("copyGR_HSV", copyGR_HSV)
	
	copyBL_HSV = hsv.copy()
	copyBL_HSV[:,:,1] = 0
	copyBL_HSV[:,:,2] = 0
	#cv2.imshow("copyBL_HSV", copyBL_HSV)
	
	
	
	cpHSV = hsv.copy()
	
	#cpHSV[:,:,0] = 0
	cpHSV[:,:,1] = 0
	#cpHSV[:,:,2] = 0
	cpH = cpHSV[...,0]
	cpS = cpHSV[...,1]
	cpV = cpHSV[...,2]
	
	hsv_image = cv2.merge([cpH, cpS, cpV])
	
	#outHSV = cv2.cvtColor(hsv_image, cv2.COLOR_LAB2BGR) # FUCKING USED THIS BEFORE
	outHSV = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
	cv2.imshow("outHSV", outHSV)

	# END TEST CODE
	####################################################################

	
	
	
	
	
	
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
	#cv2.waitKey(0)
	
	
	
	# Background, support structures, and dirt pixels
	
	
	contHSV = convertBGRHSV(contourRes.copy())
	cv2.imshow("contHSV", contHSV)
	contHSVRange = getColourRange(contHSV, lower_bg, upper_bg)
	contImgLoc = getPlantLocation(contourRes.copy(), contHSVRange)
	#cv2.imshow("contImgLoc", contImgLoc)
	
	
	dirtHSV = contHSV.copy()
	dirtHSVRange = getColourRange(dirtHSV, lower_dirt, upper_dirt)
	dirtImgLoc = getPlantLocation(contourRes.copy(), dirtHSVRange)
	#cv2.imshow("dirtImgLoc", dirtImgLoc)

	
	supportHSV = contHSV.copy()
	supportHSVRange = getColourRange(supportHSV, lower_support, upper_support)
	supportImgLoc = getPlantLocation(contourRes.copy(), supportHSVRange)
	#cv2.imshow("supportImgLoc", supportImgLoc)

	
	
	
	
	
	# Add the background, support structures, and dirt pixels together
	# Make a mask from them
	# Use that mask to remove all non-plant pixels
	# Get Width and Height data
	
	bgDirt = cv2.add(contImgLoc,dirtImgLoc)
	cv2.imshow("bgDirt", bgDirt)
	
	allNonPlant = cv2.add(bgDirt,supportImgLoc)
	cv2.imshow("allNonPlant", allNonPlant)
	
	grayNon = cv2.cvtColor(allNonPlant, cv2.COLOR_BGR2GRAY)
	
	ret, blkmask = cv2.threshold(grayNon, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)
	blkmask_inv = cv2.bitwise_not(blkmask)
	
	#cv2.imshow("blkmask", blkmask)
	#cv2.imshow("blkmask_inv", blkmask_inv)
	
	cont = contourRes.copy()
	contAnd = cv2.bitwise_and(cont, cont, mask = blkmask)
	#cv2.imshow("contAnd", contAnd)
	
	
	
	cannyContAnd = applyCanny(contAnd, 30, 200)
	#cv2.imshow("cannyContAnd", cannyContAnd)
	
	
	blur = cv2.GaussianBlur(contAnd.copy(),(5,5),0)
	#cv2.imshow("blur", blur)
	blurContAnd = applyCanny(blur, 30, 200)
	#cv2.imshow("blurContAnd", blurContAnd)
	
	
	
	bilatCont = applyBilateralFilter(contAnd.copy(), 11, 17, 17)
	bilatContAnd = applyCanny(bilatCont, 30, 200)
	#cv2.imshow("bilatContAnd", bilatContAnd)
	
	edgeHSV1 = cannyContAnd.copy()
	#edgeHSV2 = bilatContAnd.copy()
	edgeHSV2 = blurContAnd.copy()
	shapeFinal = contAnd.shape

	doubleHSVEdge = mergeEdges(edgeHSV1, edgeHSV2, shapeFinal)
	cv2.imshow("doubleHSVEdge", doubleHSVEdge)
	
	
	
	finalContour = getContoursWrap(contAnd, doubleHSVEdge)
	cv2.imshow("finalContour", finalContour)
	
	'''
	
	##############
	# FIX THIS PORTION TO FIND THE IMAGE CONTOUR
	
	#finalContour = getContoursWrap(andMInv, doubleEdge)
	#cv2.imshow("finalContour", finalContour)
	
	
	final = convertBGRHSV(andMInv)
	finalHSV = getColourRange(final, lower_green, upper_green)
	
	bilatHSV = applyBilateralFilter(andMInv.copy(), 11, 17, 17)
	filteredHSV = convertBGRHSV(bilatHSV)
	filteredHSVRange = getColourRange(filteredHSV, lower_green, upper_green)

	finalHSVLoc = getPlantLocation(andMInv.copy(), finalHSV)
	cv2.imshow("finalHSVLoc", finalHSVLoc)
	
	filteredHSVLoc = getPlantLocation(andMInv.copy(), filteredHSVRange)
	cv2.imshow("filteredHSVLoc", filteredHSVLoc)
	
	
	mergedPlantAreas = mergeImages(finalHSVLoc, filteredHSVLoc, 0.5, 0.5)
	#cv2.imshow("mergedPlantAreas", mergedPlantAreas)
	
	edgeHSVLoc = applyCanny(finalHSVLoc, 30, 200)
	edgeFilteredHSVLoc = applyCanny(filteredHSVLoc, 30, 200)



	edgeHSV1 = edgeHSVLoc.copy()
	edgeHSV2 = edgeFilteredHSVLoc.copy()
	shapeFinal = andMInv.shape

	doubleHSVEdge = mergeEdges(edgeHSV1, edgeHSV2, shapeFinal)
	#cv2.imshow("doubleHSVEdge", doubleHSVEdge)
	
	
	finalContour = getContoursWrap(andMInv.copy(), doubleHSVEdge)
	cv2.imshow("finalContour", finalContour)
	
	'''
	
	
	
	
	
	
	
	contheight, contwidth = contAnd.shape[:2]
	print("contheight:" + str(contheight) + "\n")
	print("contwidth:" + str(contwidth) + "\n")
	
	
	
	
	
	# 17 March 2018 12:26pm
	# QR Code stuff
	# https://github.com/NaturalHistoryMuseum/pyzbar
	# https://www.learnopencv.com/barcode-and-qr-code-scanner-using-zbar-and-opencv/
	
	decodedObjects = decode(plantOrig.copy())
	#disp = display(plantOrig.copy(), decodedObjects)
	#cv2.imshow('disp', disp)

	print(decodedObjects[0].data)
	
	#qr = qrcodeDimensions(plantOrig.copy(), decodedObjects)
	#cv2.imshow('qr', qr)

	qrX, qrY = qrcodeDimensions(decodedObjects)
	print("qrX:" + str(qrX) + "\n")
	print("qrY:" + str(qrY) + "\n")
	
	
	cm = 5
	
	# how Many pixels per centimetre
	cmWidth = qrX / cm
	cmHeight = qrY / cm
	
	
	print("Plant Width: " + str(contwidth / cmWidth) + "cm \n")
	print("Plant Height: " + str(contheight / cmHeight) + "cm \n")
	
	#cv2.waitKey(0)
	
	
	
	
	# www.philipzucker.com/aruco-in-opencv/

	'''
		drawMarker(...)
			drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
	'''
	''' 
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	print(aruco_dict)
	# second parameter is id number
	# last parameter is total image size
	aru = aruco.drawMarker(aruco_dict, 2, 700)
	cv2.imwrite("test_marker.jpg", aru)
	
	cv2.imshow('aru',aru)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	'''
	
	'''
	frame = plantOrig.copy()
	#print(frame.shape) #480x640

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()

	#print(parameters)
	'''
	'''    detectMarkers(...)
		detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
		mgPoints]]]]) -> corners, ids, rejectedImgPoints
		'''
		#lists of ids and the corners beloning to each id
	'''
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	print(corners)

	#It's working.
	# my problem was that the cellphone put black all around it. The alrogithm
	# depends very much upon finding rectangular black blobs
	
	gray = aruco.drawDetectedMarkers(gray, corners)
	
	#print(rejectedImgPoints)
	# Display the resulting frame
	cv2.imshow('frame',gray)
	'''
	
	if(showAll):
		#print (count)
		for i in range(count):
			cv2.imshow(processedImages[i][1], processedImages[i][0])
	cv2.waitKey(0)
	
	return contourRes, mergedPlantAreas










# STARTS HERE
# OPENS FILE / SOMEHOW GETS FILE
# FROM STORAGE, OR FROM CAMERA*
# *(Need to add camera operations, maybe)
if __name__ == '__main__':

	# Set bool to append / not append images to list
	addhsvrange = False
	addlab = False
	addlabBGR = False
	addfilteredRange = False
	addorigImgLoc = False
	addfilteredImgLoc = False

	addedgeLoc = False
	addedgeFilteredLoc = False
	adddoubleEdge = False

	#addcontour = True
	#addcontourFiltered = True
	addcontourRes = False

	# Set bool to Show all images added to list
	showAll = False

	# Number of plants in image (Can be defined by user later on)
	numberPlants = 1

	#file = easygui.fileopenbox()
	plantImg = readInPlant("PEA_16_QR_RANDOM_FLAT.png")
	#plantImg = readInPlant("PEA_16_QR_DISTORT3.png")
	#plantImg = readInPlant("PEA_18.png")
	#plantImg = readInPlant("plantqr.jpg")


	height, width = plantImg.shape[:2]
	plantImg = cv2.resize(plantImg,(1854, 966), interpolation = cv2.INTER_CUBIC)



	cv2.imshow("plantImg", plantImg)

	# Processing pipeline
	processed, pContours = process(plantImg)

	cv2.imshow("processed", processed)
	cv2.imshow('pContours', pContours)


	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	cv2.imwrite('./images/final.png', processed)


#cv2.waitKey(0)