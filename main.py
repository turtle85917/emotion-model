import cv2
import keras
import numpy
import joblib
from xgboost import XGBClassifier
from mediapipe.python.solutions import drawing_utils, face_mesh
from utils import getModelInput, CLASS

face = face_mesh.FaceMesh(
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
)
scaler = joblib.load("scaler.pkl")
# model = XGBClassifier()
# model.load_model("emotion_classification.json")
model = keras.models.load_model("emotion_classification.keras")

video = cv2.VideoCapture(0)
while video.isOpened():
  ret, img = video.read()

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.flip(img, 1)
  faceResult = face.process(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if not ret: break
  if faceResult.multi_face_landmarks is None: continue

  faceLandmarks = faceResult.multi_face_landmarks[0]

  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LIPS, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_NOSE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))

  x = getModelInput(faceLandmarks.landmark)
  pred = model.predict(scaler.transform(numpy.array([x])), verbose=0)
  maximumIdx = numpy.argmax(pred, axis=1)[0]
  print(f"{CLASS[maximumIdx]}로 예측됨 ({pred[0][maximumIdx] * 100:.2f}%)")
  # predProb = model.predict_proba(x)[0]
  # print(predProb)

  k = cv2.waitKey(30)
  if k == 27:
    break

  cv2.imshow("frame", img)

video.release()
cv2.destroyAllWindows()
