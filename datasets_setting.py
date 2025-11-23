import shutil
import os
import kagglehub
from utils import CLASS

# FER2013 = "msambare/fer2013"
RAFDB = "shuvoalok/raf-db-dataset"

print("Start setting datasets")

# Environment
download_datasets = True
affectnet_resize_images = False

# Download dataset
if download_datasets:
  if os.path.exists(".cache"):
    shutil.rmtree(".cache")
  os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), ".cache")

  # path = kagglehub.dataset_download(FER2013)
  # print(f"Downloaded dataset FER-2013 : {path}")

  path = kagglehub.dataset_download(RAFDB)
  print(f"Downloaded dataset RAF-DB : {path}")
else:
  print("Skip downloading datasets")

# # Constants
# getPath = lambda x: os.path.join(os.getcwd(), ".cache/datasets", x, "versions/1")
# KDEF_PATH_IN_CACHE = getPath(KDEF)
# AFFECTNET_PATH_IN_CACHE = os.path.join(getPath(AFFECTNET), "archive (3)/Train")

# # Merge files in each dataset folder
# shutil.rmtree("dataset")
# os.makedirs("dataset")
# print("Start merging files in each dataset folder")
# for item in CLASS:
#   os.makedirs(os.path.join("dataset", item))
# for dataset in [KDEF_PATH_IN_CACHE, AFFECTNET_PATH_IN_CACHE]:
#   print(f"I see {dataset}")
#   for item in CLASS:
#     cnt = 1
#     files = os.listdir(os.path.join(dataset, item))
#     for file in files:
#       os.rename(
#         os.path.join(dataset, item, file),
#         os.path.join("dataset", item, file)
#       )
#       print(f"[{item}] {file} 옮기기 완료")
#       cnt += 1
