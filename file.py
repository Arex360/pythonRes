import cv2
from detection import detect
faceCas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1) # try -1, 0 , 1 in the case if you can't see video
while True:
    _,img = cap.read()
    img = detect(img,faceCas)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()