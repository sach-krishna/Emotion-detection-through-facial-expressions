import cv2
from keras.models import model_from_json
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)  
video_file = cv2.VideoCapture('stimulus video.mp4')  
if not video_file.isOpened():
    print("Error: Could not open video file.")
    exit()

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

emotion_data = []

while True:
    ret_webcam, im_webcam = webcam.read()
    if not ret_webcam:
        break

    gray_webcam = cv2.cvtColor(im_webcam, cv2.COLOR_BGR2GRAY)

    faces_webcam = face_cascade.detectMultiScale(gray_webcam, 1.3, 5)

    for (p, q, r, s) in faces_webcam:
        image = gray_webcam[q:q+s, p:p+r]
        cv2.rectangle(im_webcam, (p, q), (p+r, q+s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        emotion_data.append([timestamp, prediction_label])

        cv2.putText(im_webcam, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    cv2.imshow("Webcam Output", im_webcam)

    ret_video, im_video = video_file.read()
    if ret_video:
        cv2.imshow("Video Output", im_video)
    else:
        video_file.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
video_file.release()
cv2.destroyAllWindows()

csv_path = 'emotion_data.csv'
with open(csv_path, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Predicted Emotion"]) 
    writer.writerows(emotion_data) 

print(f"Emotion data has been saved to '{csv_path}'.")

df = pd.read_csv(csv_path)

emotion_counts = df['Predicted Emotion'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Emotion Distribution')
plt.axis('equal')

plt.show()
