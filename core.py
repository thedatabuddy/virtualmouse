import cv2
import numpy as np

lowerBound=np.array([60,80,40])
upperBound=np.array([102,255,255])  #HSV Color bounds
cam=cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))   #Morphology opening kernel
kernelClose=np.ones((15,15))    #Morphology closing kernel

font=cv2.FONT_HERSHEY_SIMPLEX    #Font settings for text

while True:
    ret, img=cam.read()
    img=cv2.resize(img,(340,220))   #Resizing image for faster processing

    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)    #Convert RGB to HSV

    mask=cv2.inRange(imgHSV,lowerBound,upperBound)  #mask with no Morphology or noise screening applied

    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)   #Applying Morphology, removing outside noise being scanned from video
    maskClose=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelOpen) #removing inner noise
    maskFinal=maskOpen
    _,conts, hier =cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #Obtaining countour box for the found object
    cv2.drawContours(img,conts,-1,(0,0,255),3)      #drawing contour box on original image

    for i in range(len(conts)): #for loop for drawing the object boundaries
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#Drawing contour rectangle
        cv2.putText(img,str(i+1),(x,y+h),font,0.55,(0,255,255),1)   #labeling with contour id

    cv2.imshow("mask",mask) #Mask OP
    cv2.imshow('MorphologyOpenMask',maskOpen)
    cv2.imshow('MorphologyCloseMask',maskClose)
    cv2.imshow("cam",img)   #Cam OP
    cv2.waitKey(10)

