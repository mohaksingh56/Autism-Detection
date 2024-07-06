import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import dlib


model = load_model('./autism_detection_2.h5')

class_labels = ['autistic', 'non_autistic']

face_detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(predictor_path)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img


def predict_autistic(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence


def detect_facial_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    landmarks = []

    for face in faces:
        shape = landmark_predictor(gray, face)
        landmarks.append(shape)

    return img, faces, landmarks


def draw_feature_box(ax, points, label, color):
    x = min(p.x for p in points)
    y = min(p.y for p in points)
    w = max(p.x for p in points) - x
    h = max(p.y for p in points) - y
    rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y - 5, label, color=color, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


img_path = r'Model_Test_Image/089.jpg'
predicted_class, confidence = predict_autistic(img_path)

print(f'Predicted class: {predicted_class} with confidence: {confidence:.2f}')


img, faces, landmarks = detect_facial_features(img_path)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img_rgb)

for face, landmark in zip(faces, landmarks):

    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y - 10, 'Face', color='red', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


    draw_feature_box(ax, [landmark.part(i) for i in range(36, 42)], 'Left Eye', 'green')
    draw_feature_box(ax, [landmark.part(i) for i in range(42, 48)], 'Right Eye', 'green')

    draw_feature_box(ax, [landmark.part(i) for i in range(48, 68)], 'Mouth', 'yellow')

plt.title(f'Predicted: {predicted_class} ({confidence:.2f})')
plt.axis('off')
plt.show()