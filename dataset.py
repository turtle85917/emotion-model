import os
import cv2
from mediapipe.python.solutions import face_mesh
import utils
from utils import pickUpSpecificPointsInFace

DATASET_FOLDER = "./dataset/"

f = open("dataset.csv", "w+")
f.write('')

chunk = os.listdir(DATASET_FOLDER)

with face_mesh.FaceMesh(
  static_image_mode=True,
  refine_landmarks=True,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
) as face:
  for item in chunk:
    path = os.path.join(DATASET_FOLDER, item)
    if not os.path.isdir(path): continue
    files = os.listdir(path)
    print(f">>> {item} 디렉토리 탐색 시작")
    for file in files:
      if not file.endswith(".png") and not file.endswith(".jpg") and not file.endswith(".jpeg"): continue
      imgPath = os.path.join(path, file)
      image = cv2.imread(imgPath)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.flip(image, 1)
      face_result = face.process(image)
      if face_result.multi_face_landmarks is None or len(face_result.multi_face_landmarks) == 0:
        print(f"[{file}] 얼굴 감지 실패함")
        continue
      oneFace = face_result.multi_face_landmarks[0].landmark
      indices = [
        *pickUpSpecificPointsInFace(oneFace, utils.FACEMESH_LIPS_IDX),
        *pickUpSpecificPointsInFace(oneFace, utils.FACEMESH_LEFT_EYE_IDX),
        *pickUpSpecificPointsInFace(oneFace, utils.FACEMESH_LEFT_EYEBROW_IDX),
        *pickUpSpecificPointsInFace(oneFace, utils.FACEMESH_RIGHT_EYE_IDX),
        *pickUpSpecificPointsInFace(oneFace, utils.FACEMESH_RIGHT_EYEBROW_IDX),
        *pickUpSpecificPointsInFace(oneFace, utils.FACEMESH_NOSE_IDX),
      ]
      positions = list(map(lambda x: f"{x[0]},{x[1]},{x[2]}", indices))
      f.write(f"{','.join(positions)},{item}\n")
      print(f"[{file}] 얼굴 감지 성공!")

f.close()
