import cv2
from pygame import mixer

mixer.init()
sound = mixer.Sound('beep.wav')

body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# Read the input image
#img = cv2.imread('test.png')
cap = cv2.VideoCapture('main.mp4')

while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = body_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
        sound.play()

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()