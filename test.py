import keras
import cv2
import joblib
import numpy
import os
import random
from mediapipe.python.solutions import drawing_utils, face_mesh
from utils import getModelInput, CLASS

TEST_IMAGE_PATH = "dataset"

face = face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.5)
scaler = joblib.load("scaler.pkl")
model = keras.models.load_model("emotion_classification.keras")

while True:
  category = input("테스트할 감정의 카테고리를 입력하세요 : ")
  categoryIdx = 0
  if category.isdecimal(): categoryIdx = int(category)
  elif category in CLASS: categoryIdx = CLASS.index(category)
  elif category == "exit": break
  elif category not in CLASS or 0 > category or category >= len(CLASS):
    print("존재하지 않는 카테고리입니다.")
    continue
  files = os.listdir(os.path.join(TEST_IMAGE_PATH, CLASS[categoryIdx]))
  img = cv2.imread(os.path.join(TEST_IMAGE_PATH, CLASS[categoryIdx], files[random.randint(0, len(files))]))
  img = cv2.resize(img, (512, 512))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.flip(img, 1)
  faceResult = face.process(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if faceResult.multi_face_landmarks is None or len(faceResult.multi_face_landmarks) == 0:
    print("얼굴을 감지할 수 없습니다.")

  faceLandmarks = faceResult.multi_face_landmarks[0]
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LIPS, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_NOSE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))

  x = getModelInput(faceLandmarks.landmark)
  p = numpy.array(x).reshape((116, 3))
  center = p.mean(axis=0, keepdims=True)
  p = (p - center).reshape((348,))
  p = scaler.transform([p])[0]
  p = p.reshape((1, 116, 3))
  pred = model.predict(p, verbose=0)
  # pred = model.predict(scaler.transform(numpy.array([x])), verbose=0)
  maximumIdx = numpy.argmax(pred, axis=1)[0]
  print(f"{CLASS[maximumIdx]}로 예측됨 ({pred[0][maximumIdx] * 100:.2f}%)")

  cv2.imshow("frame", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
