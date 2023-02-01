
"""INPUT"""

#SEMANTIC SEGMENTATION ver2
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


dir = "/content/gdrive/MyDrive/Datasets/nerf_data/semantic_segmentation/cello_single"
images_train_path = os.path.join(dir,'train')
images_test_path = os.path.join(dir,'test')
images_val_path = os.path.join(dir,'val')
transform_train_path = os.path.join(dir,'transforms_train.json')
transform_test_path = os.path.join(dir,'transforms_test.json')
transform_val_path = os.path.join(dir,'transforms_val.json')


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) /255.0 #Convert to W, H, 1 dimension
        img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA) 
        img = (img != 1.0) *1.0 #Binarize
        if img is not None:
            images.append(img)
    return np.array(images)

train_imgs = load_images_from_folder(images_train_path)
test_imgs =  load_images_from_folder(images_test_path)
val_imgs =  load_images_from_folder(images_val_path)

with open(transform_train_path, "r" ) as f:
  train_transform = json.load(f)
with open(transform_test_path, "r" ) as f:
  test_transform = json.load(f)
with open(transform_val_path, "r" ) as f:
  val_transform = json.load(f)

def get_transform_data(transform, W):
  camera_angle_x =  transform['camera_angle_x']
  tform_cam2world = np.array([x['transform_matrix'] for x in train_transform['frames']])
  focal = np.array([.5 * W / np.tan(.5 * camera_angle_x)])
  return tform_cam2world, focal, camera_angle_x

plt.imshow(train_imgs[0])