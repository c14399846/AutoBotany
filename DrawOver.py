import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
#from matplotlib import image as image
import easygui

# Re-process to get ideal contours / features
class DrawOver:
	
	height = 500
	width = 500
	
	image = np.zeros((height,width,3), np.uint8)
	reference = np.zeros((height,width,3), np.uint8)
	contours = np.zeros((height,width,3), np.uint8)
	numberPlants = 0

	'''def __init__(self, image):
		cv2.imshow("lImg", image)
		cv2.waitKey(0)
	'''

	def __init__(self, image, reference, contours, numberPlants):

		self.image = reference
		self.reference = reference
		self.contours = contours
		self.numberPlants = numberPlants
	
	
	
	# Apply Canny Edge
	def applyCanny(self, image, lowerEdge, upperEdge):

		canny = cv2.Canny(image,lowerEdge, upperEdge)
		
		return canny
	
	# Merge Plant Locations
	def mergeImages(self, image1, image2, wgtImg1, wgtImg2):
		
		mergedImages = cv2.addWeighted(image1, wgtImg1, image2, wgtImg2, 0)
		
		return mergedImages
	
	
	
	def drawNew(self):
		
		print ("\nPress 'a' to add area \n")
		print ("Press 'r' to remove area \n")
		print ("Press 'i' to increase brush, 'd' to decrease \n")

		lowerB = (255, 0, 0)
		higherB = (255, 0, 0)
		
		lowerR = (0, 0, 255)
		higherR = (0, 0, 255)
		
		output = self.image.copy()
		drawHSV = self.reference.copy()
		height, width = output.shape[:2]


		
		#drawing = False # true if mouse is pressed
		ix,iy = -1, -1
		cSize = 5 # Circle size for drawing
		rWidth = 5
		rHeight = 5
		
		
		#cColour = (0,0,255) # (255,0,0)
		whiteColour = (255,255,255) 
		blackColour = (0,0,0) 

		# Holds the drawing elements
		tmpImg = np.ones((height,width,3), np.uint8)
		tmpImgBlack = np.zeros((height,width,3), np.uint8)
		tmpImgBlack[:,:] = (255,255,255)
		
		# mouse callback function
		def draw_circle_TEST(event,x,y,flags,param):
			global ix,iy,drawing
			
			#drawing = False
			
			holdingImg = self.image.copy()
			
			
			if event == cv2.EVENT_LBUTTONDOWN:
				drawing = True
				ix,iy = x,y
			
			elif event == cv2.EVENT_MOUSEMOVE:
				if drawing == True:
				
					X1 = x-rWidth
					X2 = x+rWidth
					
					Y1 = y-rHeight
					Y2 = y+rHeight
					
					if adding:
						
						cv2.rectangle(img = drawHSV, 
							pt1 = (X1,Y1), 
							pt2 = (X2,Y2), 
							color = (255,0,0), 
							thickness = -1)
						
						cv2.rectangle(img = tmpImgBlack, 
							pt1 = (X1,Y1), 
							pt2 = (X2,Y2), 
							color = whiteColour, 
							thickness = -1)
						
						tmpImg[Y1:Y2,X1:X2] = self.image[Y1:Y2,X1:X2]
						
					elif not adding:

						cv2.rectangle(img = tmpImgBlack, 
							pt1 = (X1,Y1), 
							pt2 = (X2,Y2), 
							color = blackColour, 
							thickness = -1)
			elif event == cv2.EVENT_LBUTTONUP:
				X1 = x-rWidth
				X2 = x+rWidth
				
				Y1 = y-rHeight
				Y2 = y+rHeight
				
				if adding:
					
					cv2.rectangle(img = drawHSV, 
						pt1 = (X1,Y1), 
						pt2 = (X2,Y2), 
						color = (255,0,0), 
						thickness = -1)
					
					cv2.rectangle(img = tmpImgBlack, 
						pt1 = (X1,Y1), 
						pt2 = (X2,Y2), 
						color = whiteColour, 
						thickness = -1)
					
					tmpImg[Y1:Y2,X1:X2] = self.image[Y1:Y2,X1:X2]
					
				elif not adding:

					cv2.rectangle(img = tmpImgBlack, 
						pt1 = (X1,Y1), 
						pt2 = (X2,Y2), 
						color = blackColour, 
						thickness = -1)

				drawing = False
			
			# TESTING NEW ORDER 13:40 22 Jan 2018
			
			# Colour portion
			mGrayImg = self.mergeImages(tmpImg.copy(), self.contours.copy(), 0.5, 0.5)
			tmpGray = cv2.cvtColor(mGrayImg,cv2.COLOR_BGR2GRAY)
			
			ret, mask = cv2.threshold(tmpGray, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)
			mask_inv = cv2.bitwise_not(mask)
			
			mImg = output
			andMInv = cv2.bitwise_and(mImg, mImg, mask = mask_inv)
			#cv2.imshow("andMInv", andMInv)
			
			
			# This is the contour + colour pixels
			tmpAll = andMInv.copy()
			
			# Black portion
			tmpBlackGray = cv2.cvtColor(tmpImgBlack,cv2.COLOR_BGR2GRAY)
			retB, maskB = cv2.threshold(tmpBlackGray, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)

			
			andMAll = cv2.bitwise_and(tmpImgBlack, tmpImgBlack,dst = tmpAll, mask = maskB)
			cv2.imshow("andMAll", andMAll)

			baseImg = self.mergeImages(tmpImg.copy(), self.contours.copy(), 1, 0)


			baseImg = andMAll
			
			edge = self.applyCanny(baseImg, 30, 200)
			cv2.imshow('edge', edge)

			
			# Finds contours (Have to be closed edges)
			(_,contoursEdge,_) = cv2.findContours(edge, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

			# Find largest contours by Area (Have to be closed contours)
			contoursEdge = sorted(contoursEdge, key = cv2.contourArea, reverse = True)		

			
			for i in range(self.numberPlants):

				hull = cv2.convexHull(contoursEdge[i])
				#cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
				img = cv2.drawContours(drawHSV, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
				img2 = cv2.drawContours(holdingImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
				

			cv2.imshow('holdingImg2', holdingImg)
			#cv2.imshow('baseImgPoly ', baseImg)

		
		adding = False
		
		cv2.namedWindow('drawHSV')
		
		# cColour is overwriting itself all the time
		cv2.setMouseCallback('drawHSV',draw_circle_TEST)


		while(1):
			cv2.imshow('drawHSV',drawHSV)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break

			if k == ord('i'):
				cSize += 1
				rWidth += 1
				rHeight += 1
			elif k == ord('d'):
				if cSize >= 2:
					cSize -= 1
					rWidth -= 1
					rHeight -= 1
			
			if k == ord('a'):
				adding = True			
			if k == ord('r'):
				adding = False
		
		cv2.waitKey(0)
		
		return drawHSV
