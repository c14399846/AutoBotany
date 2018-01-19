import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lower_green = (30,60,60)
upper_green = (80,255,255)




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
		
		'''
		place = i
		x,y,w,h = cv2.boundingRect(contoursEdge[i])
		cv2.rectangle(plant, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(plant,str(place),(x, (y-10)), font, 1,(255,255,255),2,cv2.LINE_AA)
		'''
		
		epsilon = 0.01*cv2.arcLength(contoursEdge[i],True)
		approx = cv2.approxPolyDP(contoursEdge[i],epsilon,True)
		
		hull = cv2.convexHull(contoursEdge[i])
		cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
		img = cv2.drawContours(baseImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 2)

	return baseImg
	
	

# The full image process pipeline
def process(plantOrig):

	kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
	processedImages = []
	count = 0
	
	
	# WORK IN PROGRESS HERE

	# Converts image to HSV colourspace
	# Gets colours in a certain range
	hsv = convertBGRHSV(plantOrig)
	hsvrange = getColourRange(hsv, lower_green, upper_green)
	if(addhsvrange):
		processedImages.append([])
		processedImages[count].append(hsvrange)
		processedImages[count].append("hsvrange")
		count += 1
	#processedImages.append(Image.open(hsvrange))
	#cv2.imshow("hsv", hsv)
	#cv2.imshow("hsvrange", hsvrange)

	
	# Applies filters to blend colours
	# *Might* make plant extraction easier
	# (for edges / contours)
	bilateral = applyBilateralFilter(plantOrig, 11, 17, 17)
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
	
	
	##########################################################
	#		TEST CODE CHANGED FROM 'labBGR' TO 'bilateral'	 #
	##########################################################
	# Convert Filtered image to HSV, get colour range for mask
	filtered = convertBGRHSV(bilateral)
	
	##########################
	# 		TEST CODE 		 #
	##########################
	#sharp = cv2.filter2D(filtered, ddepth = -1, kernel = kernelSharp)
	#cv2.imshow("sharp", sharp)
	#cv2.imshow("filt", filtered)
	#filtered = sharp
	
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
	#cv2.imshow("origImgLoc", origImgLoc)
	
	##########################################################################
	# 						TEST CODE REMOVE LATER 							 #
	##########################################################################
	#morphed = applyMorph(filteredRange)
	#filteredRange = morphed
	#cv2.imshow("morphed", morphed)
	
	
	
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
	
	
	# NOT A GOOD PLACE TO MORPH 
	'''
	morph1 = applyMorph(origImgLoc)
	morph2 = applyMorph(filteredImgLoc)
	
	cv2.imshow("orig", origImgLoc)
	cv2.imshow("morph1", morph1)
	cv2.imshow("filt", filteredImgLoc)
	cv2.imshow("morph2", morph2)
	'''
	
	# Gets Edges of Plant Pixels
	edgeLoc = applyCanny(origImgLoc, 30, 200)
	if(addedgeLoc):
		processedImages.append([])
		processedImages[count].append(edgeLoc)
		processedImages[count].append("edgeLoc")
		count += 1
	#cv2.imshow("edgeLoc", edgeLoc)
	
	
	# Gets Edges of Filtered Plant Pixels
	edgeFilteredLoc = applyCanny(filteredImgLoc, 30, 200)
	if(addedgeFilteredLoc):
		processedImages.append([])
		processedImages[count].append(edgeFilteredLoc)
		processedImages[count].append("edgeFilteredLoc")
		count += 1
	#cv2.imshow("edgeFilteredLoc", edgeFilteredLoc2)
	
	
	
	
	
	# Adds the two Canny Edges together, better coverage achieved
	#
	# https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
	# Reference code
	# NEED TO CLEAN THIS STUFF
	
	img1 = edgeLoc.copy()
	img2 = edgeFilteredLoc.copy()
	
	shape = plantOrig.shape
	
	
	# I want to put logo on top-left corner, So I create a ROI
	rows,cols,channels = shape
	roi = img1[0:rows, 0:cols ]
	# Now create a mask of logo and create its inverse mask also
	#img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	img2gray = img2.copy()
	ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	# Now black-out the area of logo in ROI
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
	# Take only region of logo from logo image.
	img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
	# Put logo in ROI and modify the main image
	dst = cv2.add(img1_bg,img2_fg)
	img1[0:rows, 0:cols ] = dst
	
	cv2.imshow('res',img1)
	
	# Finds Contours from Both Edges
	contourRes = getContours(plantOrig, img1)
	if(addcontourRes):
		processedImages.append([])
		processedImages[count].append(contourRes)
		processedImages[count].append("contourRes")
		count += 1
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
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
	#THIS ONE IS NOT GREAT, FIX THE FILTERING
	
	
	
	if(showAll):
		print (count)
		for i in range(count):
			cv2.imshow(processedImages[i][1], processedImages[i][0])
	cv2.waitKey(0)


# STARTS HERE
# OPENS FILE / SOMEHOW GETS FILE
# FROM STORAGE, OR FROM CAMERA*
# *(Need to add camera operations, maybe)

file = easygui.fileopenbox()
plantImg = readInPlant(file)
cv2.imshow("plantImg", plantImg)




# Set bool to append / not append images to list
addhsvrange = False
addlab = False
addlabBGR = False
addfilteredRange = False
addorigImgLoc = False
addfilteredImgLoc = False

addedgeLoc = True
addedgeFilteredLoc = True
addcontour = True
addcontourFiltered = True
addcontourRes = True

# Set bool to Show all images added to list
showAll = True

numberPlants = 2


# Processing pipeline
process(plantImg)
	
	
	


cv2.waitKey(0)