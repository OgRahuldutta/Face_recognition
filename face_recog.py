import cv2

# Define the function to draw boundary and recognize faces
def recognize_face(img, clf, faceCascade):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_img, 1.1, 10)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Purple rectangle (BGR: 255, 0, 255)
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))
        
        if confidence > 77:
            names = {1: "Rahul Dutta", 2: "Ankit Nayak", 3: "Umang Bhradwaj", 4: "Dishant Gogoi"}
            name = names.get(id, "Unknown")
            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)  # Red text
        else:
            cv2.putText(img, "UNKNOWN!", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)  # Red text
    
    return img

# Load face detector and classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = recognize_face(img, clf, faceCascade)
    cv2.imshow("Face Recognition", img)
    
    if cv2.waitKey(1) == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
