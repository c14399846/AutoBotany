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
		#cv2.rectangle(baseImg, (x,y), (x+w, y+h), (0,255,0), 2)
		#cv2.putText(baseImg,str(place),(x, (y-10)), font, 1,(255,255,255),2,cv2.LINE_AA)
		
		# Crops plant out of image, for later usage
		cropped = baseImg[y-5:y+h+5, x-5:x+w+5]
		#cv2.imshow("cropped", cropped)
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
	#cv2.imshow("sharpOrig", sharpOrig)
	
	# Has interesting seperation of Colours,
	# Need to get good ranges for it
	YUV = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2YUV)
	#cv2.imshow("YUV", YUV)
	
	sharpenYUV = cv2.filter2D(YUV, ddepth = -1, kernel = kernelSharp)
	#cv2.imshow("sharpenYUV", sharpenYUV)
	
	LAB = cv2.cvtColor(plantOrig, cv2.COLOR_BGR2LAB)
	#cv2.imshow("LAB", LAB)
	
	sharpenLAB = cv2.filter2D(LAB, ddepth = -1, kernel = kernelSharp)
	#cv2.imshow("sharpenLAB", sharpenLAB)
	
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
	#cv2.imshow("hsv", hsv)
	#cv2.imshow("hsvrange", hsvrange)

	sharpenHSV = cv2.filter2D(hsv, ddepth = -1, kernel = kernelSharp)
	#cv2.imshow("sharpenHSV", sharpenHSV)
	
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
	#cv2.imshow("origImgLoc", origImgLoc)
	
	
	
	# Finds Plant Pixels matching the Filtered Mask
	filteredImgLoc = getPlantLocation(plantOrig, filteredRange)
	if(addfilteredImgLoc):
		processedImages.append([])
		processedImages[count].append(filteredImgLoc)
		processedImages[count].append("filteredImgLoc")
		count += 1
	#cv2.imshow("filteredImgLoc", filteredImgLoc)
	
	
	mergedPlantAreas = mergeImages(origImgLoc, filteredImgLoc, 0.5, 0.5)
	
	
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
	
	return contourRes, mergedPlantAreas

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
#resized = cv2.resize(plantImg,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)



cv2.imshow("plantImg", plantImg)







# Processing pipeline
processed, pContours = process(plantImg)
cv2.imshow("processed", processed)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Pass original iamge, and processed image as reference
#processed = drawOver(plantImg, processed)
	
'''
TEST CODE FOR DRAWING ON IMAGES
FROM LABS WEEK 2_2

'''
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

'''






'''
TEST CODE FOR DRAWING ON IMAGES
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
'''
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
'''



# Draw on Areas you want to add / remove from the plant image
# Re-process to get ideal contours / features
def drawOver(image, reference, contours):

	#cv2.imshow(" Image", image)
	#cv2.imshow("reference Image", reference)
	#cv2.imshow("contours Image", contours)

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

	#imageRes = cv2.resize(image, (int(width/2), int(height/2)))
	#referenceRes = cv2.resize(reference, (int(width/2), int(height/2)))
	#combined = np.concatenate((imageRes, referenceRes), axis=0)
	#combined = cv2.resize(combined, (width,height))
	#cv2.imshow("combined Image", combined)
	
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
	#tmpImgBlack = np.ones((height,width,3), np.uint8)
	tmpImgBlack = np.zeros((height,width,3), np.uint8)
	tmpImgBlack[:,:] = (255,255,255)
	
	#startedDrawing = False
	
	
	
	# mouse callback function
	def draw_circle_TEST(event,x,y,flags,param):
		global ix,iy,drawing
		
		
		if event == cv2.EVENT_LBUTTONDOWN:
			#startedDrawing = True
			drawing = True
			ix,iy = x,y
		
		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
			
				X1 = x-rWidth
				X2 = x+rWidth
				
				Y1 = y-rHeight
				Y2 = y+rHeight
				
				if adding:
					#print ("add")
					
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
					#print ("remove")
					'''
					cv2.rectangle(img = drawHSV, 
						pt1 = (X1,Y1), 
						pt2 = (X2,Y2), 
						color = (0,0,255), 
						thickness = -1)
					'''
					'''
					cv2.rectangle(img = tmpImg, 
						pt1 = (X1,Y1), 
						pt2 = (X2,Y2), 
						color = blackColour, 
						thickness = -1)
					'''

					cv2.rectangle(img = tmpImgBlack, 
						pt1 = (X1,Y1), 
						pt2 = (X2,Y2), 
						color = blackColour, 
						thickness = -1)
			
				#cv2.circle(drawHSV,(x,y),cSize,cColour,-1)
				#cv2.circle(tmpImg,(x,y),cSize,blackColour,-1)
				#cv2.ellipse(tmpImg,(x,y),(cSize,cSize),0,0,360,255,-1)
				#ell = cv2.ellipse(tmpImg,(x,y),(cSize,cSize),0,0,360,255, -1)
		
				#print (ell.shape)
				
		elif event == cv2.EVENT_LBUTTONUP:
			X1 = x-rWidth
			X2 = x+rWidth
			
			Y1 = y-rHeight
			Y2 = y+rHeight
			
			if adding:
				#print ("add")
				#cColour = (255,0,0)
				
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
				#print ("remove")
			
				#cColour = (0,0,255)			
			
				'''cv2.rectangle(img = drawHSV, 
					pt1 = (X1,Y1), 
					pt2 = (X2,Y2), 
					color = (0,0,255), 
					thickness = -1)
				'''
				'''
				cv2.rectangle(img = tmpImg, 
					pt1 = (X1,Y1), 
					pt2 = (X2,Y2), 
					color = blackColour, 
					thickness = -1)
				'''
				cv2.rectangle(img = tmpImgBlack, 
					pt1 = (X1,Y1), 
					pt2 = (X2,Y2), 
					color = blackColour, 
					thickness = -1)
			
			
			
			#tmpImg[Y1:Y2,X1:X2] = image[Y1:Y2,X1:X2]
			#cv2.circle(drawHSV, (x,y), cSize, cColour, -1)
			#cv2.circle(tmpImg, (x,y), cSize, blackColour, -1)
			drawing = False

	
	
		'''
			REALLY WONKY TEST CODE.
			
			TRYING TO DO LIVE UPPDATED CONTOURING.
			
			
			# Can do more efficently, remove the range stuff, just set the baseimg circles to black or
			# The original image pixel area.
			#
			# Show the user the updating image, but use a scratch image for processing changes.
			#
			
		'''
		
		'''
		roiB = cv2.inRange(tmpImg, lowerB, higherB)
		roiR = cv2.inRange(tmpImg, lowerR, higherR)
		
		corrected = cv2.bitwise_and(output, output, mask = roiB)
		
		mask_inv_b = cv2.bitwise_not(roiB)
		mask_inv_r = cv2.bitwise_not(roiR)
		corrected = cv2.bitwise_or(output, output, mask = mask_inv_r)
		'''
		
		
		#edgeImg = contours.copy()
		
		'''
		'mask' is picking up the 'removing' rectangles, but not the addition ones
		Show up as white squares in 'mask' imshow
		***************MY THRESHOLDING IS BORKED******************FIX MEEEEEEEEEE
		'''
		
		
		
		
		
		'''
		JUST SAVE THE ADD AND REMOVE TO SEPERATE IMAGES
		THEN PROCEED TO OVERLAY THEM IN DIFFERENT ORDERS
		
		# put contours over the added portions, then put removed over the contours portion
		# This could work (Similar to extracting tmpImg pixels)
		# https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
		# Use this link, might help
		
		
		
		'''
		
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
		#cv2.imshow("tmpAll", tmpAll)
		
		# Black portion
		tmpBlackGray = cv2.cvtColor(tmpImgBlack,cv2.COLOR_BGR2GRAY)
		retB, maskB = cv2.threshold(tmpBlackGray, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)
		maskB_inv = cv2.bitwise_not(maskB)
		#andMInv = cv2.bitwise_and(mImg, mImg, mask = maskB_inv)
		
		
		# **************************************************************************************************
		#
		# THIS ONE IS CLOSE 14:01 22 JAN 2018
		# IT DOESNT WANT TO OVERWRITE WITH COLOUR THOUGH
		# tmpAll is the issue, because of copy() and andMInv or w/e
		
		# 14:11 22 JAN 2018
		
		'''
		
		The issue is obvious
		the black rects are not being removed or overwritten on, they are the last layer being placed onto the image stackoverflow
			ergo: They can't have colour pixels placed on top of them
		
		'''
		
		andMAll = cv2.bitwise_and(tmpImgBlack, tmpImgBlack,dst = tmpAll, mask = maskB)
		cv2.imshow("andMAll", andMAll)
		#
		# **************************************************************************************************
		
		#cv2.imshow("tmpBlackGray", tmpBlackGray)
		#cv2.imshow("maskB", maskB) # white rectangles on black BG
		#cv2.imshow("maskB_inv", maskB_inv) # black rectangles on white BG
		
		
		
		
		
		# Black is not covering the plant pixels
		# Go back and think about the actual pizel gathering process I've made*********
		# 22-Jan-2018 11:30am 
		mGrayImg = mergeImages(tmpImg.copy(), contours.copy(), 0.5, 0.5)
		#cv2.imshow("tmpImg", tmpImg)
		#cv2.imshow("contours", contours)
		#cv2.imshow("mGrayImg", mGrayImg)
		
		
		tmpGray = cv2.cvtColor(mGrayImg,cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(tmpGray, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY_INV)
		#ret, mask = cv2.threshold(tmpGray, thresh = 0, maxval = 1, type = cv2.THRESH_BINARY_INV) 
		# 22 Jan 2018 11:07 better threshold
		
		# Orig threshold
		#ret, mask = cv2.threshold(tmpGray, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)
		mask_inv = cv2.bitwise_not(mask)
		#cv2.imshow("tmpGray", tmpGray)
		#cv2.imshow("mask", mask)
		#cv2.imshow("mask_inv", mask_inv)
		
		mImg = output
		
		#mImg = mergeImages(tmpImg.copy(), contours.copy(), 0.5, 0.5)
		
		# output vs contours.copy()
		
		# Shows whole image and Removed portions
		andM = cv2.bitwise_and(mImg, mImg, mask = mask)
		orM = cv2.bitwise_or(mImg, mImg, mask = mask)
		#cv2.imshow("andM", andM)
		#cv2.imshow("orM", orM)
		
		
		# Only shows added portions
		# This one is good? I think?
		andMInv = cv2.bitwise_and(mImg, mImg, mask = mask_inv)
		orMInv = cv2.bitwise_or(mImg, mImg, mask = mask_inv)
		#cv2.imshow("andMInv", andMInv)
		#cv2.imshow("orMInv", orMInv)
		
		
		
		# THIS THING IS NOT BEING CORRECLT OVERWRITTEN
		# BLACK COLOURING IS NOT WORKING
		baseImg = mergeImages(tmpImg.copy(), contours.copy(), 1, 0)

		
		#baseImg = mergeImages(tmpImg.copy(), edgeImg, 0.1, 0.9)
		#baseImg2 = mergeImages(tmpImg.copy(), edgeImg, 0.3, 0.7)
		#baseImg3 = mergeImages(tmpImg.copy(), edgeImg, 0.5, 0.5)
		#baseImg4 = mergeImages(tmpImg.copy(), edgeImg, 0.7, 0.3)
		#baseImg5 = mergeImages(tmpImg.copy(), edgeImg, 0.9, 0.1)
		
		
		'''
		cv2.imshow("baseImg", baseImg)
		cv2.imshow("baseImg2", baseImg2)
		cv2.imshow("baseImg3", baseImg3)
		cv2.imshow("baseImg4", baseImg4)
		cv2.imshow("baseImg5", baseImg5)'''
		#cv2.imshow("tmpImg", tmpImg)
		#cv2.imshow("contours", contours)
		
		#cv2.waitKey(0)
		
		
		
		# THIS IS WHY CONTOURS WERE NOT WORKING CORRECTLY************************
		# IT WAS NOT GETTING MERGED AT ALL,
		# JUST USING THE BASE CONTOURS COPY**************************************
		# 
		# CONTOUR USES 'baseImg' FOR GETTING POLYLINES, AND 'edge' FOR CANNY
		# SO THE RESULTS ARE SCREWED UP
		#
		
		edge = applyCanny(baseImg, 30, 200)

		#cv2.imshow("baseImg", baseImg)
		#cv2.imshow("edge", edge)
		
		
		# Finds contours (Have to be closed edges)
		(_,contoursEdge,_) = cv2.findContours(edge, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

		# Find largest contours by Area (Have to be closed contours)
		contoursEdge = sorted(contoursEdge, key = cv2.contourArea, reverse = True)		
		
		
		
		'''
		
		NEED TO PASS IN A TEMP IMAGE (NOT 'drawHSV')
		
		WHY?
			BECAUSE IT DOESNT RESET THE CONTOURS,
			IT ONLY DRAWS OVER THEM,
			SO IT'S A MESS
		
		
		
		'''
		for i in range(numberPlants):
			
			# Draw rectangles, with order of Contour size
			
			#place = i
			#x,y,w,h = cv2.boundingRect(contoursEdge[i])
			
			
			hull = cv2.convexHull(contoursEdge[i])
			cv2.polylines(baseImg, pts=hull, isClosed=True, color=(0,255,255))
			img = cv2.drawContours(drawHSV, contoursEdge[i], contourIdx=-1, color=(0,0,255), thickness = 1)
			
			#print ("success")

		
		'''
		END WONKY TEST CODE
		'''
		#cv2.imshow('tmp', tmpImg)

	
	
	
	
	
	
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
				rWidth += 1
				rHeight += 1
		
		if k == ord('a'):
			adding = True
			#cColour = (255,0,0)
			
		if k == ord('r'):
			adding = False
			#cColour = (0,0,255)
		
	#cv2.destroyAllWindows()
	#cv2.waitKey(0)

	#tmpGray = cv2.cvtColor(tmpImg,cv2.COLOR_BGR2GRAY)
	#ret, mask = cv2.threshold(tmpGray, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)
	#mask_inv = cv2.bitwise_not(mask)
	
	
	
	# THIS IS FIALING BECAUSE IM NOT MERGING THE IMAGES TOGETHER CORRECTLY ~12:25pm 20 Jan 2018
	# STILL KINDA SORTA FAILING, 12:47pm 20 Jan 2018
	
	'''
	lowerB = (255, 0, 0)
	higherB = (255, 0, 0)
	
	lowerR = (0, 0, 255)
	higherR = (0, 0, 255)
	
	
	
	
	# convert to HSV, then get range
	#converted = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2HSV)

	#cv2.imshow("tmpImg", tmpImg)
	#cv2.imshow("converted", converted)
	
	roiB = cv2.inRange(tmpImg, lowerB, higherB)
	roiR = cv2.inRange(tmpImg, lowerR, higherR)
	#cv2.imshow("roiB", roiB)
	#cv2.imshow("roiR", roiR)
	#cv2.imshow("output", output)
	
	corrected = cv2.bitwise_and(output, output, mask = roiB)
	#cv2.imshow("Corr B", corrected)
	
	mask_inv_b = cv2.bitwise_not(roiB)
	mask_inv_r = cv2.bitwise_not(roiR)
	#cv2.imshow("mask_inv B", mask_inv_b)
	#cv2.imshow("mask_inv R", mask_inv_r)
	
	corrected = cv2.bitwise_or(output, output, mask = mask_inv_r)
	#cv2.imshow("Corr R", corrected)
	
	#corrected = cv2.bitwise_or(corrected, output, mask = roiR)
	#cv2.imshow("Corr R", corrected)
	
	
	# Final image, with empty variable for the returned contour value (not needed right now)
	final, _ = process(corrected)
	
	#cv2.imshow("final", final)
	
	'''
	
	
	cv2.waitKey(0)
	
	return

# Global bool, because, reasons
drawing = False

# Pass original image, and processed image as reference
processed = drawOver(plantImg, processed, pContours)

cv2.imshow("redrawn", processed)	



'''

drawHSV = plantImg.copy()

drawing = False # true if mouse is pressed
ix,iy = -1,-1
cSize = 5 # Circle size for drawing


# Holds the drawing elements
tmpImg = np.zeros((height,width,3), np.uint8)
startedDrawing = False

# mouse callback function
def draw_circle(event,x,y,flags,param):
	global ix,iy,drawing
	
	if event == cv2.EVENT_LBUTTONDOWN:
		startedDrawing = True
		drawing = True
		ix,iy = x,y
	
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			cv2.circle(drawHSV,(x,y),cSize,(0,0,255),-1)
			cv2.circle(tmpImg,(x,y),cSize,(0,0,255),-1)
	
	elif event == cv2.EVENT_LBUTTONUP:
		cv2.circle(drawHSV, (x,y), cSize, (0, 0, 255), -1)
		cv2.circle(tmpImg, (x,y), cSize, (0, 0, 255), -1)
		drawing = False

#img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('drawHSV')
cv2.setMouseCallback('drawHSV',draw_circle)



while(1):
    cv2.imshow('drawHSV',drawHSV)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
	
	#if k == ord('i'):
	#	cSize += 1
	
	#if k == ord('d'):
	#	if(cSize >= 2):
	#		cSize = cSize-1

    if k == ord('i'):
        cSize += 1
    elif k == ord('d'):
        if cSize >= 2:
            cSize -= 1
	
cv2.destroyAllWindows()
cv2.waitKey(0)

tmpGray = cv2.cvtColor(tmpImg,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(tmpGray, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)


#cv2.imshow("tmpImg", tmpImg)
#cv2.imshow("mask", mask)	
cv2.imshow("mask_inv", mask_inv)	
cv2.waitKey(0)
'''




'''
TEST CODE
# GETS THE ROI FROM DRAG AND DROP MOUSE CLICKS
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

'''	


'''
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
		
		
'''	
		
		
		
		
		
		
		
		
		
		
		
cv2.waitKey(0)