import cv2
import keras
import numpy
import joblib
import math
from xgboost import XGBClassifier
from mediapipe.python.solutions import drawing_utils, face_detection, face_mesh
from utils import getModelInput, getRealPoint, CLASS

face = face_mesh.FaceMesh(
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
)
faceDetection = face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
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
  faceDetectionResult = faceDetection.process(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if not ret: break
  if faceResult.multi_face_landmarks is None: continue
  if faceDetectionResult.detections is None: continue
  if len(faceDetectionResult.detections) == 0: continue

  faceLandmarks = faceResult.multi_face_landmarks[0]
  oneOfFaceDetectionResult = faceDetectionResult.detections[0]

  positions = getModelInput(faceLandmarks.landmark)
  # x = numpy.array([positions])
  # x = scaler.fit_transform(x)
  p = numpy.array(positions).reshape((116, 3))
  center = p.mean(axis=0, keepdims=True)
  p = (p - center).reshape((348,))
  p = scaler.transform([p])[0]
  p = p.reshape((1, 116, 3))
  pred = model.predict(p, verbose=0)
  predZip = [(pred[0][i], i) for i in range(len(pred[0]))]
  predZip.sort(key=lambda x: x[0])

  # text = "\n".join(f"{CLASS[i]} : {pred[0][i] * 100:.8f}" for i in range(len(pred[0])))
  # maximumIdx = numpy.argmax(pred, axis=1)[0]
  # print(f"{CLASS[maximumIdx]}로 예측됨 ({pred[0][maximumIdx] * 100:.2f}%)")
  # predProb = model.predict_proba(x)[0]
  # print(predProb)
  # drawing_utils.draw_detection(img, oneOfFaceDetectionResult)
  # print(oneOfFaceDetectionResult)

  box = oneOfFaceDetectionResult.location_data.relative_bounding_box
  heightCenter = getRealPoint((box.height * 0.5 + box.ymin), img.shape[0])
  halfCount = len(predZip) // 2

  for i in range(len(predZip)):
    text = f"{CLASS[predZip[i][1]]} : {predZip[i][0] * 100:.8f}"
    resultText = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
    cv2.putText(img, text=text, org=(10 + getRealPoint((box.width + box.xmin), img.shape[1]), heightCenter + (halfCount - i) * resultText[0][1] + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255) if i == len(predZip) - 1 else (0, 0, 0), thickness=1)

  cv2.rectangle(img,
    (getRealPoint(box.xmin, img.shape[1]), getRealPoint(box.ymin, img.shape[0])),
    (getRealPoint((box.xmin + box.width), img.shape[1]), getRealPoint((box.ymin + box.height), img.shape[0])),
    (0, 0, 255), 2
  )
  # drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LIPS, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  # drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  # drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  # drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  # drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  # drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_NOSE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))

  k = cv2.waitKey(30)
  if k == 27:
    break

  cv2.imshow("frame", img)

face.close()
faceDetection.close()
video.release()
cv2.destroyAllWindows()
