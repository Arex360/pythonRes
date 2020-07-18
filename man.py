import cv2
from detection import detect
img = cv2.imread('man.jpg')
img = detect(img,faceCas) 
cv2.imshow('sample image',img)
 
cv2.waitKey(0) 
cv2.destroyAllWindows()