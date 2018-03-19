
# Draw on Areas you want to add / remove from the plant image
# Re-process to get ideal contours / features
def drawOver(image, reference, contours):

	print ("\nPress 'a' to add area \n")
	print ("Press 'r' to remove area \n")
	print ("Press 'i' to increase brush, 'd' to decrease \n")

	lowerB = (255, 0, 0)
	higherB = (255, 0, 0)
	
	lowerR = (0, 0, 255)
	higherR = (0, 0, 255)
	
	output = image.copy()
	drawHSV = reference.copy()
	height, width = output.shape[:2]

	ix,iy = -1, -1
	cSize = 5 # Circle size for drawing
	rWidth = 5
	rHeight = 5
	
	
	whiteColour = (255,255,255) 
	blackColour = (0,0,0) 

	# Holds the drawing elements
	tmpImg = np.ones((height,width,3), np.uint8)
	tmpImgBlack = np.zeros((height,width,3), np.uint8)
	tmpImgBlack[:,:] = (255,255,255)
	
	
	
	tmpImgWhite = np.zeros((height,width,3), np.uint8)
	tmpImgWhite[:,:] = (255,255,255)

	
	# mouse callback function
	def draw_circle_TEST(event,x,y,flags,param):
		global ix,iy,drawing
		
		
		holdingImg = image.copy()
		
		
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
					
					tmpImg[Y1:Y2,X1:X2] = image[Y1:Y2,X1:X2]
					
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
				
				tmpImg[Y1:Y2,X1:X2] = image[Y1:Y2,X1:X2]
				
			elif not adding:

				cv2.rectangle(img = tmpImgBlack, 
					pt1 = (X1,Y1), 
					pt2 = (X2,Y2), 
					color = blackColour, 
					thickness = -1)
			
			drawing = False

		
		# TESTING NEW ORDER 13:40 22 Jan 2018
		
		# Colour portion
		mGrayImg = mergeImages(tmpImg.copy(), contours.copy(), 0.5, 0.5)
		tmpGray = cv2.cvtColor(mGrayImg,cv2.COLOR_BGR2GRAY)
		
		#tmpAll = tmpGray.copy()
		
		ret, mask = cv2.threshold(tmpGray, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)
		mask_inv = cv2.bitwise_not(mask)
		
		mImg = output
		andMInv = cv2.bitwise_and(mImg, mImg, mask = mask_inv)
		cv2.imshow("andMInv", andMInv)
		
		
		# This is the contour + colour pixels
		tmpAll = andMInv.copy()
		
		# Black portion
		tmpBlackGray = cv2.cvtColor(tmpImgBlack,cv2.COLOR_BGR2GRAY)
		retB, maskB = cv2.threshold(tmpBlackGray, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)
		maskB_inv = cv2.bitwise_not(maskB)
		
		
		andMAll = cv2.bitwise_and(tmpImgBlack, tmpImgBlack,dst = tmpAll, mask = maskB)	

		
		# Black is not covering the plant pixels
		# Go back and think about the actual pizel gathering process I've made*********
		# 22-Jan-2018 11:30am 
		mGrayImg = mergeImages(tmpImg.copy(), contours.copy(), 0.5, 0.5)
		
		
		tmpGray = cv2.cvtColor(mGrayImg,cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(tmpGray, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)
		#ret, mask = cv2.threshold(tmpGray, thresh = 0, maxval = 1, type = cv2.THRESH_BINARY_INV) 
		# 22 Jan 2018 11:07 better threshold
		
		# Orig threshold
		mask_inv = cv2.bitwise_not(mask)

		
		mImg = output
		
		# output vs contours.copy()
		
		# Shows whole image and Removed portions
		andM = cv2.bitwise_and(mImg, mImg, mask = mask)
		orM = cv2.bitwise_or(mImg, mImg, mask = mask)
		
		
		# Only shows added portions
		# This one is good? I think?
		andMInv = cv2.bitwise_and(mImg, mImg, mask = mask_inv)
		orMInv = cv2.bitwise_or(mImg, mImg, mask = mask_inv)
		
		
		
		# THIS THING IS NOT BEING CORRECLT OVERWRITTEN
		# BLACK COLOURING IS NOT WORKING
		baseImg = mergeImages(tmpImg.copy(), contours.copy(), 1, 0)
		
		# USED TO GET CANNY EDGES***********************
		baseImg = andMAll
		
		edge = applyCanny(baseImg, 30, 200)		
		
		# Finds contours (Have to be closed edges)
		(_,contoursEdge,_) = cv2.findContours(edge, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

		# Find largest contours by Area (Have to be closed contours)
		contoursEdge = sorted(contoursEdge, key = cv2.contourArea, reverse = True)		
		
		for i in range(numberPlants):
			
			# Draw rectangles, with order of Contour size
			
			hull = cv2.convexHull(contoursEdge[i])
			img = cv2.drawContours(drawHSV, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
			
			# TESTING UPDATED CONTOURS
			img2 = cv2.drawContours(holdingImg, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
		
		#tmpImgWhite
		tmpHSV = cv2.cvtColor(tmpAll,cv2.COLOR_BGR2GRAY)
		retHSV, maskHSV = cv2.threshold(tmpHSV, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)
		
		maskHSV_inv = cv2.bitwise_not(maskHSV)
		tstHSV = cv2.bitwise_and(output, output, mask = maskHSV_inv)
	
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

# Global bool, because, reasons
drawing = False





# Pass original image, and processed image as reference


# Internal Python code
#processed = drawOver(plantImg, processed, pContours)


# Python code in external file
#dOver = DrawOver(plantImg, processed, pContours, numberPlants)
#processed = dOver.drawNew()





#cv2.imshow("redrawn", processed)
#cv2.imshow("redrawn", processed)	