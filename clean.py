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
	

# The full image process pipeline
def process(plantOrig):

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
	
	
	# Convert Filtered image to HSV, get colour range for mask
	filtered = convertBGRHSV(labBGR)
	filteredRange = getColourRange(filtered, lower_green, upper_green)
	if(addfilteredRange):
		processedImages.append([])
		processedImages[count].append(filteredRange)
		processedImages[count].append("filteredRange")
		count += 1
		#cv2.imshow("addfilteredRange", addfilteredRange)
	
	
	origImgLoc = getPlantLocation(plantOrig, hsvrange)
	if(addorigImgLoc):
		processedImages.append([])
		processedImages[count].append(origImgLoc)
		processedImages[count].append("origImgLoc")
		count += 1
	
	
	filteredImgLoc = getPlantLocation(plantOrig, filteredRange)
	if(addfilteredImgLoc):
		processedImages.append([])
		processedImages[count].append(filteredImgLoc)
		processedImages[count].append("filteredImgLoc")
		count += 1
	
	#cv2.imshow("origImgLoc", origImLoc)
	#cv2.imshow("filteredImgLoc", filteredImgLoc)
	
	
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
addorigImgLoc = True
addfilteredImgLoc = True

# Set bool to Show all images added to list
showAll = True




# Processing pipeline
process(plantImg)
	
	
	


cv2.waitKey(0)