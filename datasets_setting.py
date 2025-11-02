import shutil
import os
import kagglehub
import cv2
from utils import CLASS

KDEF = "chenrich/kdef-database"
AFFECTNET = "mstjebashazida/affectnet"

print("Start setting datasets")

# Environment
download_datasets = False
affectnet_resize_images = False

# Download dataset
if download_datasets:
  shutil.rmtree(".cache")
  os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), ".cache")

  path = kagglehub.dataset_download(KDEF)
  print(f"Downloaded dataset KDEF : {path}")

  path = kagglehub.dataset_download(AFFECTNET)
  print(f"Downloaded dataset AffectNet : {path}")
else:
  print("Skip downloading datasets")

# Constants
getPath = lambda x: os.path.join(os.getcwd(), ".cache/datasets", x, "versions/1")
KDEF_PATH_IN_CACHE = getPath(KDEF)
AFFECTNET_PATH_IN_CACHE = os.path.join(getPath(AFFECTNET), "archive (3)/Train")

# Resize images in Affect
if affectnet_resize_images:
  angerPath = os.path.join(AFFECTNET_PATH_IN_CACHE, "anger")
  if os.path.exists(angerPath):
    os.rename(angerPath, os.path.join(AFFECTNET_PATH_IN_CACHE, "angry"))
  maxCount, count = 0, 0
  for _, _, files in os.walk(AFFECTNET_PATH_IN_CACHE):
    maxCount += len(files)
  prevLen = 0
  for item in CLASS:
    files = os.listdir(os.path.join(AFFECTNET_PATH_IN_CACHE, item))
    for file in files:
      if not file.endswith(".png") and not file.endswith(".jpg") and not file.endswith(".jpeg"): continue
      filePath = os.path.join(AFFECTNET_PATH_IN_CACHE, item, file)
      img = cv2.imread(filePath)
      img = cv2.resize(img, (512, 512))
      cv2.imwrite(filePath, img)
      text = f"[{item}] {file} : 이미지 사이즈 변경 완료! ({count} / {maxCount})"
      print(f"\r{text}{' ' * max(0, prevLen - len(text))}", end='')
      prevLen = len(text); count += 1
  print()
else:
  print("Skip resizing images in AffectNet")

# Merge files in each dataset folder
shutil.rmtree("dataset")
os.makedirs("dataset")
print("Start merging files in each dataset folder")
for item in CLASS:
  os.makedirs(os.path.join("dataset", item))
for dataset in [KDEF_PATH_IN_CACHE, AFFECTNET_PATH_IN_CACHE]:
  print(f"I see {dataset}")
  for item in CLASS:
    cnt = 1
    files = os.listdir(os.path.join(dataset, item))
    for file in files:
      os.rename(
        os.path.join(dataset, item, file),
        os.path.join("dataset", item, file)
      )
      print(f"[{item}] {file} 옮기기 완료")
      cnt += 1
