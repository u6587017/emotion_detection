This project implements a computer vision pipeline that detects human faces in a video stream, classifies their emotional states, and provides a per-frame tally of the emotions present.

## Tools:
- OpenCV: Used for video stream handling, image preprocessing, and face detection (typically via Haar Cascades or DNN modules).
- Keras/MobileNet: A lightweight Deep Learning architecture used to classify detected faces into specific emotion categories from 28709 images for training set and 7178 images for validation belonging to 7 classes.
- {'angry': 0,
 'disgust': 1,
 'fear': 2,
 'happy': 3,
 'neutral': 4,
 'sad': 5,
 'surprise': 6}
<br />
Parameters:
steps_per_epoch=10, 
epochs=30,
EarlyStopping (patience=5),
optimizer='adam',
loss= categorical_crossentropy , metrics=['accuracy'] 
<br />

## Example training images:
<img width="425" height="435" alt="example_input" src="https://github.com/user-attachments/assets/a4468a89-8862-499d-bc05-b834e24b829f" /> <img width="425" height="435" alt="example_input2" src="https://github.com/user-attachments/assets/3ec81662-e43f-48fa-b634-a2bbb4ca3bc9" />
<br />


### Accuracy:
Blue: accuracy, Red:val_accuracy
<br />
<img width="547" height="435" alt="acc" src="https://github.com/user-attachments/assets/4b4aa23e-51cf-4c18-9a3f-366fbfc9d15a" />
<br />

### Validation:
Blue: loss, Red:val_loss
<br />
<img width="547" height="435" alt="loss" src="https://github.com/user-attachments/assets/2b5ff131-271f-43b8-9655-d886238c2730" />

## Trump video test:
<img width="1073" height="950" alt="Screenshot 2026-03-09 221500" src="https://github.com/user-attachments/assets/f091413e-c42b-4d3e-94fa-fc10653ab35e" />
<br />
<img width="1074" height="941" alt="Screenshot 2026-03-09 221511" src="https://github.com/user-attachments/assets/08d77ce3-0c24-42b1-bf31-878ef8233ce4" />
<br />
<img width="1057" height="819" alt="Screenshot 2026-03-09 221548" src="https://github.com/user-attachments/assets/fa48cec4-b5fa-4ec7-966f-4903a2e28280" />
