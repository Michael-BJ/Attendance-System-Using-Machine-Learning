# Attendance System using Machine Learning

How it's works ?
---
1. First install libary that we need
````
pip install opencv-python   
pip install numpy
pip install face-recognition
pip install pyttsx3 
pip install DateTime
````

2. Capture the face that you want to recognize 
````
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('take a picture ',frame)
    if cv2.waitKey(1) & 0xFF == ord('y') :
        cv2.imwrite('my_picture.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()
````
3. Connect the program to your webcam 
````
import cv2
import numpy as numpy

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
`````

4. Load the image that you want to recognize 
````
etcodetech_image = face_recognition.load_image_file("etcodetech.jpg")
etcodetech_face_encoding = face_recognition.face_encodings(etcodetech_image)[0]

known_face_encoding = [
    etcodetech_face_encoding
]
known_face_names = [
    "etcodetech"
]
````
5. Download the model of face and load to the program
````
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
````
6. For identification the picture or video change into gray 
````
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3,5)
````
7. Change BGR to RGB
````
 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
````
8. Compare the known face and unknown face
````
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
````

9. Put the text (name) and rectangle to the picture or video
````
for ((x, y, w, h), name) in zip(faces, names):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
    0.75, (0, 255, 0), 2)`
````
10. Create a csv file which we will connect to the program later

11. Make a program that can insert the output to the csv file
````
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
````
11. connect the code to the speaker 
````
if matches [0] == True:
    engine.say("Good Morning, please go to your class" ) 
    makeAttendanceEntry(name)
else:
    engine.say("Please try again, i don't recognize you")
engine.runAndWait() 
````