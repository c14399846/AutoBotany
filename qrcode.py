
# CODE DERVIED FORM THESE SOURCES, WITH ADAPTATIONS
# Need to get 'dll' files to have this working correctly

# https://github.com/NaturalHistoryMuseum/pyzbar
# https://www.learnopencv.com/barcode-and-qr-code-scanner-using-zbar-and-opencv/
# https://github.com/bharathp666/opencv_qr
# aishack.in/tutorials/scanning-qr-codes-verify-finder/


# Random string generator
# https://www.random.org/strings/?num=10&len=25&digits=on&upperalpha=on&loweralpha=on&unique=on&format=html&rnd=new
# YfxhPDD3SR9QJobJM08G
# qrcode_random_id.png


from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
 
def decode(im) : 

  height, width = im.shape[:2]
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  #cv2.imshow('gray', gray)
  #cv2.waitKey(0)

  # Find barcodes and QR codes  
  decodedObjects = pyzbar.decode((gray.tobytes(), width, height))
 
  # Print results
  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data,'\n')
	
  return decodedObjects
 

# Display barcode and QR code location  
def display(im, decodedObjects):
 
  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.rect
 
    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else : 
      hull = points;
     
    # Number of points in the convex hull
    n = len(hull)
    print(n)
    print(hull)
 
    # Draw the convext hull
    #for p in range(0,n):
    #for j in range(0,n):
      #cv2.polylines(im, pts=hull, isClosed=True, color=(0,255,255))
      #cv2.drawContours(im, hull, contourIdx=-1, color=(0,0,255), thickness = 1)
      #cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
      #cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
	  
    X = hull[0]
    width = hull[2]
    Y = hull[1]
    height = hull[3]
    
    cv2.line(im, (X,Y), (X + width,Y), (255,0,0), 3) # top line
    cv2.line(im, (X,Y), (X, Y + height), (255,0,0), 3) # left line
    cv2.line(im, (X,Y + height), (X + width, Y + height), (255,0,0), 3) # bottom line
    cv2.line(im, (X + width, Y + height), (X + width, Y), (255,0,0), 3) # right line
    #cv2.drawContours(im, hull, contourIdx=-1, color=(0,0,255), thickness = 1)

  # Display results
  cv2.imshow("Results", im);
  cv2.waitKey(0);
 


# NB*********************
# DISTORT3 has a 'lens distortion' filter applied at main 25
# anything higher really messes up qr code tracking
# So this is not an exact science
# Main 
if __name__ == '__main__':
 
  # Read image
  #im = cv2.imread('qrcode.png')
  #im = cv2.imread('PEA_16_QR.png')
  #im = cv2.imread('PEA_16_QR_DISTORT3.png')
  im = cv2.imread('google.png')
  #im = cv2.imread('plant1.png')
 
  decodedObjects = decode(im)
  display(im, decodedObjects)