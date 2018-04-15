from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import sys
import numpy as np
import cv2
from matplotlib import image as image
import os
import time

'''
# Need to have these installed on your machine
#
# pip install pyzbar matplotlib numpy
#
# Note:
#	Make sure you start your Anaconda instance if you installed OpenCV in an Anaconda module
#	[e.g]
# 
#		workon cv
#	
#		(cv) python imageProcess.py <FILEPATH> <FILENAME>
#		(cv) python imageProcess.py './plantImage.png' 'plantImage'
#
'''


# Set bool to append / not append images to list
adddetectedPlant = False
addlab = False
addlabBGR = False
adddetectedFilteredPlant = False
addorigImgLoc = False
addfilteredImgLoc = False
addedgeLoc = False
addedgeFilteredLoc = False
adddoubleEdge = False
addcontourRes = False
addcontAnd = False

# Set bool to Show all images added to list
showAll = False



# Number of plants in image (Can be defined by user later on)
numberPlants = 1



# Read in plant image
def readInPlant(imagePath):

	plant = cv2.imread(imagePath)

	return plant



# Convert BGR to GRAY
def convertBGRGray(image):

	grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	return grayImg

	

# Convert Gray to BGR
def convertGray2BGR(image):

	bgrImg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	return bgrImg

	

# Converts BGR to HSV
def convertBGRHSV(image):

	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	return HSV



# Converts HSV to BGR
def convertHSVBGR(image):

	BGR = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

	return BGR



# Get Threshold of image
def getThreshold(image, lowerTH, upperTH):

	_, thresh = cv2.threshold(image, lowerTH, upperTH, cv2.THRESH_BINARY)

	return thresh



# Returns colours that match the colourspace range
def getColourRange(image, lower, upper):
	
	range = cv2.inRange(image, lower, upper)
	
	return range



# Seperates colours in image that match the mask
def getPlantLocation(image, range):

	plantLocation = cv2.bitwise_and(image.copy(), image, mask = range)
	
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

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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



# Apply Canny Edge
def applyCanny(image, lowerEdge, upperEdge):

	canny = cv2.Canny(image,lowerEdge, upperEdge)
	
	return canny
	

	
# Apply Morphological Processes
def applyMorph(image):

	largeMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
	smallMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

	image = cv2.dilate(image,largeMorph)

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
	if(contoursEdge):
		for i in range(numberPlants):

			x,y,w,h = cv2.boundingRect(contoursEdge[i])

			# Cropping is here (If it works.....)
			baseImg = baseImg[y:y+h, x:x+w]

		
	return baseImg




# Find the QR Code in the image
def decode(im) : 

	height, width = im.shape[:2]
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Find barcodes and QR codes  
	decodedObjects = pyzbar.decode((gray.tobytes(), width, height))
 
	# Print results
	'''
	for obj in decodedObjects:
		print('Type : ', obj.type)
		print('Data : ', obj.data,'\n')
	'''
	return decodedObjects


# Display the QR Code on the image passed into the function
# Display barcode and QR code location  
def display(im, decodedObjects):

	# Loop over all decoded objects
	for decodedObject in decodedObjects:
	
		points = decodedObject.rect
		#print(points)

		# If the points do not form a quad, find convex hull
		if len(points) > 4 : 
			hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
			hull = list(map(tuple, np.squeeze(hull)))
		else : 
			hull = points;

		# Number of points in the convex hull
		n = len(hull)

		# X and Y positions,
		# and the distance between X2 and Y2 (width + height)
		X = hull[0]
		width = hull[2]
		Y = hull[1]
		height = hull[3]

		# Draw lines around the QR Code
		cv2.line(im, (X,Y), (X + width,Y), (255,0,0), 1) # top line
		cv2.line(im, (X,Y), (X, Y + height), (255,0,0), 1) # left line
		cv2.line(im, (X,Y + height), (X + width, Y + height), (255,0,0), 1) # bottom line
		cv2.line(im, (X + width, Y + height), (X + width, Y), (255,0,0), 1) # right line

	return im


# Get Dimensions of the QR Code in the image
# Return Width and Height data
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


	return width, height


# Extract plantID data from the QR Code
# Convert it to String format
def qrcodeGetPlantID(decodedObjects):
	
	ID = -1;
	
	for obj in decodedObjects:
		ID = obj.data

	# Convert from bytearray to String
	finalID = ID.decode("utf-8")


	return finalID

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
	
	if(contoursEdge):
		for i in range(numberPlants):
			
			# Contours that wrap around the plant object
			hull = cv2.convexHull(contoursEdge[i])
			cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
			img = cv2.drawContours(baseImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
			
	return baseImg
	
	
	
def removeBackground(contourRes):
	
	# HSV background data colour ranges
	# Used to identify the pixel data
	
	# Background / Wall ranges
	lower_bg  = (20, 70, 200)
	upper_bg = (33, 150, 255) # This one rmeoved a lot of crap, the plant look decent too

	# Dirt ranges
	lower_dirt = (20, 130, 25)
	upper_dirt = (30, 255, 115)

	# Support Structure ranges
	lower_support = (19, 67, 70)
	upper_support = (30, 252, 253)
	
	# END HSV colour ranges

	
	# YCB colour ranges
	lower_ycb = (60, 128, 94)
	upper_ycb = (150, 145, 100)

	# Find location of the background in the iamge
	contHSV = convertBGRHSV(contourRes)
	bgHSVRange = getColourRange(contHSV, lower_bg, upper_bg)
	bgImgLoc = getPlantLocation(contourRes, bgHSVRange)
	
	
	# Find location of dirt in the image
	dirtHSV = contHSV.copy()
	dirtHSVRange = getColourRange(dirtHSV, lower_dirt, upper_dirt)
	dirtImgLoc = getPlantLocation(contourRes, dirtHSVRange)

	
	# Find location of support structures in the image
	supportHSV = contHSV.copy()
	supportHSVRange = getColourRange(supportHSV, lower_support, upper_support)
	supportImgLoc = getPlantLocation(contourRes, supportHSVRange)

	
	
	# Merge the background, support structures, and dirt pixels together
	# Make a mask from them
	# Use that mask to remove all non-plant pixels
	# Get Width and Height data
	
	bgDirtLoc = cv2.add(bgImgLoc,dirtImgLoc)
	
	allNonPlantLoc = cv2.add(bgDirtLoc,supportImgLoc)
	
	# Use YCB to extract Support Structures a bit better
	# (The plants reflects green light onto the support structures too much)
	# (This causes the parts of the support to be seen as part of the plant)
	YCB = cv2.cvtColor(contourRes, cv2.COLOR_BGR2YCrCb)
	ycbrange = getColourRange(YCB, lower_ycb, upper_ycb)
	ycbSupportLoc = getPlantLocation(contourRes, ycbrange)
	ycbnon = cv2.add(allNonPlantLoc,ycbSupportLoc)
	
	
	grayNon = convertBGRGray(ycbnon)
	ret, graymask = cv2.threshold(grayNon, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)
	#graymask_inv = cv2.bitwise_not(graymask)

	# This is the original retruned value
	contour = cv2.bitwise_and(contourRes, contourRes, mask = graymask)
	
	
	# Edge detecton for better contours
	'''
	cannyContAnd = applyCanny(contour, 30, 200)
	
	# Use blurred image for better edge overlap
	blur = cv2.GaussianBlur(contour.copy(),(5,5),0)
	blurContAnd = applyCanny(blur, 30, 200)
	
	
	edgeHSV1 = cannyContAnd.copy()
	edgeHSV2 = blurContAnd.copy()
	shapeFinal = contour.shape

	doubleHSVEdge = mergeEdges(edgeHSV1, edgeHSV2, shapeFinal)
	
	

	# Displays Contour as a red line over the plant image
	finalContour = getContoursWrap(contour, doubleHSVEdge)
	#cv2.imshow("finalContour", finalContour)
	
	contheight, contwidth = contour.shape[:2]
	#print("contheight:" + str(contheight) + "\n")
	#print("contwidth:" + str(contwidth) + "\n")
	'''
	return contour

	
	
	
def detectPlant(detPlant):

	# HSV colour range to find 'Green' plants
	# I.E, pea plant
	lower_green = (30,60,60) # Lower Plant Colourspace
	upper_green = (80,255,255)	# Upper Plant Colourspace

	hsv = convertBGRHSV(detPlant)
	hsvrange = getColourRange(hsv, lower_green, upper_green)

	return hsvrange
	
	

# The full image process pipeline
def process(plantOrig):

	processedImages = [] # Array of processed images
	count = 0 			 # Integer that holds number of images processed + saved
	
	kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
	kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)


	

	# Using CLAHE
	grayCLA = convertBGRGray(plantOrig.copy())

	lab = cv2.cvtColor(plantOrig.copy(), cv2.COLOR_BGR2LAB)
	planes = cv2.split(lab)
	planes[0] = applyCLAHE(planes[0])
	lab = cv2.merge(planes)
	cla = convertLABBGR(lab)

	#cv2.imshow('orig', plantOrig)
	#cv2.imshow('cla', cla)
	#cv2.waitKey()
	
	plantOrig = cla

	# Converts image to HSV colourspace
	# Gets colours in a certain range
	
	detectedPlant = detectPlant(plantOrig)
	
	if(adddetectedPlant):
		processedImages.append([])
		processedImages[count].append(detectedPlant)
		processedImages[count].append("detectedPlant")
		count += 1
	#cv2.imshow("detectedPlant", detectedPlant)
	
	
	# Finds Plant Pixels matching the Mask
	origImgLoc = getPlantLocation(plantOrig, detectedPlant)
	if(addorigImgLoc):
		processedImages.append([])
		processedImages[count].append(origImgLoc)
		processedImages[count].append("origImgLoc")
		count += 1
	#cv2.imshow("origImgLoc", origImgLoc)
	
	
	
	# Applies filters to blend colours
	# *Might* make plant extraction easier (for edges / contours)
	bilateral = applyBilateralFilter(plantOrig, 11, 17, 17)

	# Convert Filtered image to HSV, get colour range for mask
	detectedFilteredPlant = detectPlant(bilateral)
	
	if(adddetectedFilteredPlant):
		processedImages.append([])
		processedImages[count].append(detectedFilteredPlant)
		processedImages[count].append("detectedFilteredPlant")
		count += 1
	#cv2.imshow("detectedFilteredPlant", detectedFilteredPlant)
	
	
	# Finds Plant Pixels matching the Filtered Mask
	filteredImgLoc = getPlantLocation(plantOrig, detectedFilteredPlant)
	if(addfilteredImgLoc):
		processedImages.append([])
		processedImages[count].append(filteredImgLoc)
		processedImages[count].append("filteredImgLoc")
		count += 1
	#cv2.imshow("filteredImgLoc", filteredImgLoc)
	
	
	# Merge two images together, tried to do edge detection, not great
	#mergedPlantAreas = mergeImages(origImgLoc, filteredImgLoc, 0.5, 0.5)
	
	
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
	# https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
	# Reference code
	
	edge1 = edgeLoc.copy()
	edge2 = edgeFilteredLoc.copy()
	shape = plantOrig.shape
	
	# Merge the two edge images together to create overlap
	doubleEdge = mergeEdges(edge1, edge2, shape)
	if(adddoubleEdge):
		processedImages.append([])
		processedImages[count].append(doubleEdge)
		processedImages[count].append("doubleEdge")
		count += 1
	#cv2.imshow("doubleEdge", doubleEdge)
	
	
	# Finds Contours from Both Edges
	contourImage = getContours(plantOrig, doubleEdge)
	if(addcontourRes):
		processedImages.append([])
		processedImages[count].append(contourImage)
		processedImages[count].append("contourImage")
		count += 1
	#cv2.imshow("contourImage", contourImage)
	
	
	
	# Extracts Background, Support structures, and Dirt pixels
	# from the contour image
	
	# Normally Machine Learning object detection would be used here, 
	# but had to manually remove the Background and etc.
	# Not enough training data to detect a plant over time
	contourCleaned = removeBackground(contourImage.copy())
	if(addcontAnd):
		processedImages.append([])
		processedImages[count].append(contourCleaned)
		processedImages[count].append("contourCleaned")
		count += 1
	#cv2.imshow("contourCleaned", contourCleaned)

	# QR Code stuff
	# Used this library to extract QR Code
	# https://github.com/NaturalHistoryMuseum/pyzbar
	# https://www.learnopencv.com/barcode-and-qr-code-scanner-using-zbar-and-opencv/
	
	decodedObjects = decode(plantOrig.copy())
	
	plantMeasurements = [];
	
	plantID = -1;
	

	contourHeight, contourWidth = contourCleaned.shape[:2]

	
	if decodedObjects is not None and len(decodedObjects) > 0:
	
		# Display QR Code location on the plant image
		#disp = display(plantOrig.copy(), decodedObjects)
		#cv2.imshow('disp', disp)

		qrX, qrY = qrcodeDimensions(decodedObjects)
		#print("qrX:" + str(qrX) + "\n")
		#print("qrY:" + str(qrY) + "\n")
		
		plantID = qrcodeGetPlantID(decodedObjects)
		
		# QR Code is 2cm by 2cm
		qr_size = 2
		
		# how Many pixels per centimetre
		qrWidth = qrX / qr_size
		qrHeight = qrY / qr_size

		cmWidth = contourWidth / qrWidth
		cmHeight = contourHeight /qrHeight

		plantMeasurements = [cmWidth, cmHeight];

		#print("Plant Width: " + str(contwidth / cmWidth) + "cm \n")
		#print("Plant Height: " + str(contheight / cmHeight) + "cm \n")
	
	
	# Used for debugging, shows all images on screen before returning an outputContoursFile
	# Press any key to continue
	if(showAll):
		for i in range(count):
			cv2.imshow(processedImages[i][1], processedImages[i][0])
	
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	return contourImage, contourCleaned, plantMeasurements, plantID




# Main,
# Accepts two arguments: [filepath, filename]
def main(filepath, filename):	

	plantImg = None

	while not os.path.exists(filepath):
		time.sleep(1)

	if os.path.isfile(filepath):
		plantImg = cv2.imread(filepath)
	else:
		raise ValueError("Not a file!")
	


	if plantImg is not None:
	
		height, width = plantImg.shape[:2]
		
		
		# Processing pipeline
		processed, pContours, pMeasurements, plantID = process(plantImg)

		#cv2.imshow("processed", processed)
		#cv2.imshow('pContours', pContours)
		
		directory = './images/'
		
		origName = filename
		
		#origPlant = 'orig.png'
		imgName = "processed_" + origName
		imgContoursName = "pContours_" + origName
		
		#origFile = directory + origPlant
		outputFile = directory + imgName
		outputContoursFile = directory + imgContoursName
		
		#cv2.imwrite(origFile, plantImg)
		cv2.imwrite(outputFile, processed)
		cv2.imwrite(outputContoursFile, pContours)
		
		print("Image saved")
		print(directory)
		print(imgName)
		print(imgContoursName)

		if(len(pMeasurements) == 2) :
			print(plantID)
			print(pMeasurements[0])
			print(pMeasurements[1])
		else :
			print("")
			print("")
			print("")
		

		#sys.exit(0)
		
	else :
		print("No Image given")
		#sys.exit(0)




# STARTS HERE
# OPENS FILE / GETS FILE
#
# How to run from command line
# python imageProcess.py <FILE LOCATION> <FILENAME>
# (e.g) python imageProcess.py './plantImage.png' 'plantImage'
#
if __name__ == '__main__':

	main(sys.argv[1], sys.argv[2])
