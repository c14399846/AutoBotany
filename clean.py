import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lower_green = (30,60,60)
upper_green = (80,255,255)


#plantImg = cv2.imread("PEA_10.png")

##################################################################################
#DONE
#####
'''s = cv2.cvtColor(plant, cv2.COLOR_BGR2HSV)
_, s_thresh = cv2.threshold(s, 85, 255, cv2.THRESH_BINARY)

s_thresh = cv2.cvtColor(s_thresh, cv2.COLOR_HSV2BGR)
s_thresh = cv2.cvtColor(s_thresh, cv2.COLOR_BGR2GRAY)
'''

#cv2.imshow('saturation', s)
#cv2.imshow('saturation_thresh', s_thresh)

# Used this for some help
# http://www.linoroid.com/2016/12/detect-a-green-color-object-with-opencv/
# Simple example stuff (the lower / higher ranges)
# 29 Dec 19:21

#################################################################################
#DONE
#####
'''
lower_green = (30,60,60)
upper_green = (80,255,255)

hsvrange = cv2.inRange(s, lower_green, upper_green)
cv2.imshow('hsvrange', hsvrange)

#Finding plants with this colour range
plantLoc = cv2.bitwise_and(pea, pea, mask = hsvrange)

cv2.imshow('plantLoc', plantLoc)
'''


####################################################################################
#DONE
#####
'''
bilateral = cv2.bilateralFilter(plantImg, 11, 17, 17)
'''
####################################################################################
#DONE
#####
'''
LAB = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(LAB)
#cv2.imshow('l_channel', l)
#cv2.imshow('a_channel', a)
#cv2.imshow('b_channel', b)
'''


####################################################################################
#DONE
#####
'''
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)
'''

####################################################################################
#DONE
#####
'''limg = cv2.merge((cl,a,b))'''
#cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
####################################################################################
#DONE
#####
'''smoothedImg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

grayImg = cv2.cvtColor(smoothedImg, cv2.COLOR_BGR2GRAY)

cv2.imshow("smoothedImg", smoothedImg)

'''


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
def getPlantMask(image, range):

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
	
	mergedLAB = cv2.merge((first, second, third))

	return mergedLAB



# Converts LAB image to BGR
def convertLABBGR(image):

	bgr = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

	return bgr


# The full image process pipeline
def process(plantOrig):

# WORK IN PROGRESS HERE
'''
	# Converts image to HSV colourspace
	# Gets colours in a certain range
	hsv = convertBGRHSV(plantOrig)
	hsvrange = getColourRange(hsv)
	
	#threshold = getThreshold(hsvrange, 85, 255)
	
	bilateral = applyBilateralFilter(hsvrange, 11, 17, 17)
	contrastSmoothed = applyCLAHE(bilateral)
'''	
	

	
# STARTS HERE
# OPENS FILE / SOMEHOW GETS FILE
# FROM STORAGE, OR FROM CAMERA*
# *(Need to add camera operations, maybe)

file = easygui.fileopenbox()
plantImg = readInPlant(file)

processed = process(plantImg)
	
	
	
	
	
	


cv2.waitKey(0)