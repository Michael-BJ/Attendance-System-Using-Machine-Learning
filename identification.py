import face_recognition
import cv2
import numpy as np
import pyttsx3
from datetime import datetime

# insert to csv file 
def makeAttendanceEntry(name):
    with open('list.csv','r+') as FILE:
        allLines = FILE.readlines()
        attendanceList = []
        for line in allLines:
            entry = line.split(',')
            attendanceList.append(entry[0])
        if name not in attendanceList:
            now = datetime.now()
            dtString = now.strftime('%d/%b/%Y, %H:%M:%S')
            FILE.writelines(f'\n{name},{dtString}')

# For sound
engine = pyttsx3.init() 

frame = cv2.imread('',cv2.IMREAD_GRAYSCALE)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

etcodetech_image = face_recognition.load_image_file("EtCodeTech.jpg")
etcodetech_face_encoding = face_recognition.face_encodings(bryan_image)[0]

known_face_encoding = [
    etcodetech_face_encoding,

known_face_names =["Etcodetech"]

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3,5)
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_face_encoding,
        encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                best_match_index = np.argmin(matches)
                name = known_face_names[best_match_index]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
            if matches [0] == True:
                engine.say("Good Morning, please go to your class" ) 
                makeAttendanceEntry(name)
            else:
                engine.say("Please try again, i don't recognize you")
            engine.runAndWait()       
    cv2.imshow("absent", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()