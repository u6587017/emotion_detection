import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from ffpyplayer.player import MediaPlayer

emotion_count = {}

# Load the trained model
model_best = load_model('detect/face_model.h5') # set your machine model file path here

# Classes 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier('detect/haarcascade_frontalface_default.xml')

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture("VIDEO.mp4")
# player = MediaPlayer("PREMIUM.NO.ADS.mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # audio_frame, val = player.get_frame()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face image to the required input size for the model
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        # Predict emotion using the loaded model
        predictions = model_best.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]

        # Display the emotion label on the frame
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        
        # print emotion count
        if emotion_label not in emotion_count:
            emotion_count[emotion_label] = 0
        else:
            emotion_count[emotion_label] += 1
        print(emotion_count)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'e' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
    # if val != 'eof' and audio_frame is not None:
    # #audio
    #     img, t = audio_frame

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()