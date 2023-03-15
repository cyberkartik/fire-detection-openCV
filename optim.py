import cv2
from playsound import playsound
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fires = fire_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
        print("Fire detected")
        playsound('audio.mp3')

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('f'):
        break
cap.release()
cv2.destroyAllWindows()