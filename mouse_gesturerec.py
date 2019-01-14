import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

pinchFlag=0
mouse=Controller()  #Mouse object

app=wx.App(False)   #initializing app to get display coordinates only
(sx,sy)=wx.GetDisplaySize() #Display Monitor Coordinates
(camx,camy)=(320,240)   #Cam resolution

mLocOld=np.array([0,0])
mouseLoc=np.array([0,0])
DampingFactor=4     #Make mouse move less violent


lowerBound=np.array([40,80,40])
upperBound=np.array([102,255,255])  #HSV Color bounds

cam=cv2.VideoCapture(0)
# cam.set(3,camx) #3 is width flag
# cam.set(4,camy) #4 is height flag
kernelOpen=np.ones((5,5))   #Morphology opening kernel
kernelClose=np.ones((8,8))    #Morphology closing kernel


while True:
    ret, img=cam.read()
    img=cv2.resize(img,(320,240))   #Resizing image for faster processing

    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)    #Convert RGB to HSV

    mask=cv2.inRange(imgHSV,lowerBound,upperBound)  #mask with no Morphology or noise screening applied

    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelClose)   #Applying Morphology, removing outside noise being scanned from video
    maskClose=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelClose) #removing inner noise
    maskFinal=maskOpen
    _,conts, hier =cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #Obtaining countour box for the found object
    cv2.drawContours(img,conts,-1,(0,0,255),3)      #drawing contour box on original image

    if(len(conts)==2):  #case for two rectangles
        if pinchFlag==1:
            pinchFlag=0
            mouse.release(Button.left)  #reducing mutiple clicks
        x1,y1,w1,h1=cv2.boundingRect(conts[0])
        x2,y2,w2,h2=cv2.boundingRect(conts[1])
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)  # Drawing contour rectangle for 1st box
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)  # Drawing contour rectangle for 1st box
        cx1, cy1 = int((x1 + w1 / 2)), int((y1 + h1 / 2))
        cx2, cy2 = int((x2 + w2 / 2)), int((y2 + h2 / 2))       #Finding middle point of both contours
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)   #Drawing a line between two contours
        cx,cy=int((cx1+cx2)/2),int((cy1+cy2)/2)
        cv2.circle(img,(cx,cy),2,(0,0,255),2)   #plotting a center point between two lines
        mouseLoc=mLocOld+((cx,cy)-mLocOld)/DampingFactor
        mouse.position=(sx-(mouseLoc[0]*sx/camx), mouseLoc[1]*sy/camy)
        mLocOld=mouseLoc

    elif(len(conts)==1):   #case for a single rectangle
        if pinchFlag==0:
            pinchFlag=1
            mouse.press(Button.left)    #reducing mutiple clicks
        x,y,w,h=cv2.boundingRect(conts[0])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cx=int(x+w/2)
        cy=int(y+h/2)
        cv2.circle(img,(cx,cy),int((w+h)/4),(0,0,255),2)    #plotting a big circle with centre as rectangle's centre
        mouseLoc = mLocOld + ((cx, cy) - mLocOld) / DampingFactor
        mouse.position = (sx - (mouseLoc[0] * sx / camx), mouseLoc[1] * sy / camy)
        mLocOld = mouseLoc

    cv2.imshow("cam",img)   #Cam OP
    cv2.waitKey(5)

