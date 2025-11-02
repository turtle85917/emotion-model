import cv2
from mediapipe.python.solutions import drawing_utils, face_mesh

face = face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.5)
img = cv2.imread(".cache/datasets/mstjebashazida/affectnet/versions/1/archive (3)/Train/sad/image0000219.jpg")
img = cv2.resize(img, (512, 512))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.flip(img, 1)
faceResult = face.process(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
if faceResult.multi_face_landmarks is None or len(faceResult.multi_face_landmarks) == 0:
  print("얼굴을 감지할 수 없습니다. 쟈넨")
else:
  faceLandmarks = faceResult.multi_face_landmarks[0]
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LIPS, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_LEFT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_RIGHT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  drawing_utils.draw_landmarks(img, faceLandmarks, face_mesh.FACEMESH_NOSE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
  cv2.imshow("test", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
