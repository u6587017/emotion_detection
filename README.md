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
EarlyStopping (patience=5)
<br />
## Example training data:
<img width="257" height="264" alt="example_input" src="https://github.com/user-attachments/assets/e0072786-b5be-49a4-9196-53916d51922a" /> <img width="257" height="264" alt="example_input2" src="https://github.com/user-attachments/assets/cbe3eac2-0a9c-4e02-9dfc-8a79ea8b2db9" />
<br />


### Accuracy:
<br />
<img width="378" height="264" alt="acc" src="https://github.com/user-attachments/assets/f335d1d2-a230-454c-b225-a292fe157515" />
<br />

### Validation:
<br />
<img width="378" height="264" alt="loss" src="https://github.com/user-attachments/assets/156e3d1c-566a-4c26-8824-3a1a33ad5097" />

## Trump video test:
<img width="1073" height="950" alt="Screenshot 2026-03-09 221500" src="https://github.com/user-attachments/assets/f091413e-c42b-4d3e-94fa-fc10653ab35e" />
<br />
<img width="1074" height="941" alt="Screenshot 2026-03-09 221511" src="https://github.com/user-attachments/assets/08d77ce3-0c24-42b1-bf31-878ef8233ce4" />
<br />
<img width="1057" height="819" alt="Screenshot 2026-03-09 221548" src="https://github.com/user-attachments/assets/fa48cec4-b5fa-4ec7-966f-4903a2e28280" />
