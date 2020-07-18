import cv2
faceCas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def drawBoundery(img, classifier,sFactor,minNab,col,text):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # We will convert the image to grayScale so that we may extract the features
    feature = classifier.detectMultiScale(gray,sFactor,minNab) # This will extract all features from the grayScalre image and weill return coords of image
    coord = []
    for (x,y,w,h) in feature:
        cv2.rectangle(img,(x,y),(x+w,y+h),col,2) # this will draw the rectangle covering up the face
        cv2.putText(img,text,(x,y-4), cv2.FONT_HERSHEY_PLAIN,1,col,1,cv2.LINE_AA)
        coord = [x,y,w,h]
    return coord , img 
def detect(img,casscade):
    color = (0,0,255)
    coords, img = drawBoundery(img,casscade,1.2,10,color,'face')
    return img
img = cv2.imread('man.jpg')
img = detect(img,faceCas) 
cv2.imshow('sample image',img)
 
cv2.waitKey(0) 
cv2.destroyAllWindows()