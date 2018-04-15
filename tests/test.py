from __future__ import division
from __future__ import print_function
from matplotlib import image as image

import cv2
import numpy as np
import os, errno
import sys
import pyzbar.pyzbar as pyzbar
import unittest

import imageProcess as imgProc

import time

'''
def testTime():
	startTime = time.time()
	endTime = time.time()
	print ("\n")
	time = str("%.2f" % round(endTime - startTime,2))
	print(time + " Seconds\n")
'''


class TestImage(unittest.TestCase):


	def setUp(self):
		self.imageLocation = './test.png'
		self.read = imgProc.readInPlant(self.imageLocation)
		pass

		
	
	def tearDown(self):
		pass
		
	
	
	def test_read_image(self):
		
		readImg = imgProc.readInPlant(self.imageLocation)

		self.assertEqual(readImg is not None, True)
	
	
	
	def test_bgr2gray_image(self):

		gray = imgProc.convertBGRGray(self.read)
		
		self.assertEqual(gray is not None, True)

		
	
	def test_bgr2hsv_image(self):

		hsv = imgProc.convertBGRHSV(self.read)
		
		self.assertEqual(hsv is not None, True)


		
	def test_hsv2bgr_image(self):

		hsv = imgProc.convertBGRHSV(self.read)
		
		bgr = imgProc.convertHSVBGR(hsv)
		
		self.assertEqual(bgr is not None, True)

	
	
	def test_bgr2lab_image(self):

		lab = imgProc.convertBGRLAB(self.read)

		self.assertEqual(lab is not None, True)

	
		
	def test_lab2bgr_image(self):

		lab = imgProc.convertBGRLAB(self.read)
		
		bgr = imgProc.convertLABBGR(lab)

		self.assertEqual(bgr is not None, True)
	
	

	def test_getthreshold_image(self):

		gray = imgProc.convertBGRGray(self.read)
		
		thresh = imgProc.getThreshold(gray, 1 , 255)
		
		self.assertEqual(thresh is not None, True)
	
	
	
	def test_getcolourrange_image(self):
	
		lower_green = (30,60,60)   # Lower Plant Colourspace
		upper_green = (80,255,255) # Upper Plant Colourspace

		hsv = imgProc.convertBGRHSV(self.read)
		
		colour = imgProc.getColourRange(hsv, lower_green, upper_green)
		
		self.assertEqual(colour is not None, True)


	
	def test_getplantloc_image(self):

		detectedPlant = imgProc.detectPlant(self.read)
		
		plantLocation = imgProc.getPlantLocation(self.read, detectedPlant)

		self.assertEqual(plantLocation is not None, True)


	
	def test_mergeimages_image(self):

		readcopy = self.read.copy()
		
		merged = imgProc.mergeImages(self.read,readcopy, 0.5, 0.5)
		
		self.assertEqual(merged is not None, True)
		
	
	
	def test_applybilateral_image(self):

		bilateral = imgProc.applyBilateralFilter(self.read, 11, 17, 17)

		self.assertEqual(bilateral is not None, True)
		

	
	def test_applyclahe_image(self):

		gray = imgProc.convertBGRGray(self.read)
		
		clahe = imgProc.applyCLAHE(gray)

		self.assertEqual(clahe is not None, True)

	


	def test_applycanny_image(self):

		gray = imgProc.convertBGRGray(self.read)
		
		ret, mask = cv2.threshold(gray, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)

		bitwisedImage = cv2.bitwise_and(self.read, self.read, mask = mask)
		
		canny = imgProc.applyCanny(bitwisedImage, 30, 200)
		
		self.assertEqual(canny is not None, True)

	

	def test_mergeedges_image(self):

		gray = imgProc.convertBGRGray(self.read)
		
		ret, mask = cv2.threshold(gray, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)

		bitwisedImage = cv2.bitwise_and(self.read, self.read, mask = mask)
		
		canny = imgProc.applyCanny(bitwisedImage, 30, 200)
		
		
		smoothedImage = cv2.GaussianBlur(self.read,(5,5),0)
		cannySmooth = imgProc.applyCanny(smoothedImage, 30, 200)
		
		mergedEdges = imgProc.mergeEdges(canny, cannySmooth, self.read.shape)
		
		self.assertEqual(mergedEdges is not None, True)
	
	'''
	# Not used
	# Splits lab into its individual channels [l, a, b]
	def splitLAB(image):

	# Merge Colour channels
	def mergeColourspace(first, second, third):

	# Merge LAB colour channels
	def mergeLAB(l, a, b):
	
	
	# Apply Morphological Processes
	def applyMorph(image):

	'''
	
class TestPerformance(unittest.TestCase):
	
	def setUp(self):
		self.imageLocation = './test.png'
		self.imageName = 'test.png'
		self.read = imgProc.readInPlant(self.imageLocation)
		pass
		
	
	def tearDown(self):
		pass
	
	
	
	def test_speedrun_image(self):
	
		startTime = time.time()
		
		imgProc.main(self.imageLocation, self.imageName)

		endTime = time.time()
		
		runtime = str("%.2f" % round(endTime - startTime,2))
		
		print(runtime + " Seconds to run Performance Test.\n")

		os.remove('./images/processed_test.png')
		os.remove('./images/pContours_test.png')
		
	
if __name__ == '__main__':
	
	print("Python environment information: ")
	print("\t" + sys.version + "\n")
	print("OpenCV Version: " + cv2.__version__ + "\n")
	
	unittest.main()
	

	
	
	
	
