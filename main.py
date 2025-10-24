import cv2
from mediapipe.python.solutions import drawing_utils, face_mesh

face = face_mesh.FaceMesh(
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
)

video = cv2.VideoCapture(0)
while video.isOpened():
  ret, img = video.read()

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.flip(img, 1)
  face_result = face.process(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if not ret: break

  if face_result.multi_face_landmarks is not None:
    for face_landmarks in face_result.multi_face_landmarks:
      drawing_utils.draw_landmarks(img, face_landmarks, face_mesh.FACEMESH_LIPS, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
      drawing_utils.draw_landmarks(img, face_landmarks, face_mesh.FACEMESH_LEFT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
      drawing_utils.draw_landmarks(img, face_landmarks, face_mesh.FACEMESH_LEFT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
      drawing_utils.draw_landmarks(img, face_landmarks, face_mesh.FACEMESH_RIGHT_EYE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
      drawing_utils.draw_landmarks(img, face_landmarks, face_mesh.FACEMESH_RIGHT_EYEBROW, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))
      drawing_utils.draw_landmarks(img, face_landmarks, face_mesh.FACEMESH_NOSE, None, drawing_utils.DrawingSpec(color=(0, 0, 255)))

  k = cv2.waitKey(30)
  if k == 27:
    break

  cv2.imshow("frame", img)

video.release()
cv2.destroyAllWindows()
