# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tqdm.notebook import tqdm
import numpy as np
import cv2
from model import init_model, render_rays_segment, get_rays
from nerf_ultils import get_avg_PSNR, visualize
from dataset import train_imgs, val_imgs, test_imgs, train_transform, test_transform, val_transform, get_transform_data
import random
import wandb


def run(train_data, 
        test_data,
        model = None,
        channel = 1,
        L_embed = 6,
        D = 8,
        W = 64,
        ray_samples = 64,
        epochs = 5000,
        see = 10,
        near = 0.0,
        far = 2.0,
        psnr_samples = 10,
        to_save = False,
        limit = None,
        lr = 5e-4):
  #UNPACK 
  (train_imgs, train_transform) = train_data
  (test_imgs, test_transform) = test_data

  #SET UP PARAMETERS
  height, width= train_imgs[0].shape[:2]
  train_poses, focal, _ = get_transform_data(train_transform, width)
  test_poses, _, _ = get_transform_data(test_transform, width)
  test_img, test_pose = test_imgs[0], test_poses[0]

  num_samples = train_imgs.shape[0]
  
  import random
  if limit:
    picked = random.sample(range(1, num_samples), limit)
    images = train_imgs[picked]
    poses = train_poses[picked]
  else:
    images = train_imgs
    poses = train_poses
  
  if not(model):
    model = init_model(D = D, W = W, L_embed = L_embed)
  optimizer = tf.keras.optimizers.Adam(lr)
  psnrs = []
  iternums = []
  training_losses = []
  valid_iternums = []
  valid_losses = []

  #TRAINING
  import time
  t = time.time()
  i_plot = epochs // see
  for i in range(epochs+1):
      img_i = np.random.randint(images.shape[0])
      target = train_imgs[img_i]
      pose = poses[img_i]
      rays_o, rays_d = get_rays(height, width, focal, pose)
      rays_o = tf.cast(rays_o, tf.float32) 
      rays_d = tf.cast(rays_d, tf.float32) 
      with tf.GradientTape() as tape:
        predicted, _, _ , _  = render_rays_segment(model, rays_o, rays_d, near=near, far=far, N_samples=ray_samples, rand=True, L_embed = L_embed) 
        loss = tf.reduce_mean(tf.square(predicted - target))
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      if i%50 ==0:
        psnr = get_avg_PSNR(model, test_data, sample_num = psnr_samples, near = near, far = far, ray_samples = ray_samples, L_embed = L_embed)
        psnrs.append(psnr)
        iternums.append(i)
        training_losses.append(loss)
  #VISUALIZE WHILE RUNNING - It shows total of "see"-parameter images, 
      if i%i_plot==0:
          print(i, (time.time() - t) / i_plot, 'secs per iter')
          img_i = np.random.randint(images.shape[0])
          img_i = 0 #FIX TO 0
          target = train_imgs[img_i]
          pose = poses[img_i]
          t = time.time()
          rays_o, rays_d = get_rays(height, width, focal, pose)
          rays_o = tf.cast(rays_o, tf.float32) #edit
          rays_d = tf.cast(rays_d, tf.float32) #edit
          predicted, _, _, _ = render_rays_segment(model, rays_o, rays_d, near=near, far=far, N_samples=ray_samples, L_embed = L_embed, channel = 4)
          loss = tf.reduce_mean(tf.square(predicted - target))
          valid_losses.append(loss)
          valid_iternums.append(i)
          fig = visualize(target, predicted, i, iternums, valid_iternums, psnrs, to_save = to_save, training_losses = training_losses, valid_losses = valid_losses)
          images = wandb.Image(fig, caption="target, predicted, psnrs, loss")
  print('Done')
  return model, focal, psnrs, training_losses, valid_losses, iternums, valid_iternums





if __name__ == "__main__":
  wandb.init(project="NeRF")
  
  # SET UP PARAMETERS  
  NEAR = 0.0
  FAR = 2.0
  CHANNEL = 4
  DEPTH = 8
  WIDTH_NETWORK = 64
  EPOCHS = 5000
  SEE = EPOCHS //50
  RAY_SAMPLES = 64
  L_EMBED = 6
  PSNR_SAMPLES = 10
  LIMIT = None
  LR  = 5e-3
  TRAIN_DATA = (train_imgs, train_transform)
  TEST_DATA = (test_imgs, test_transform)

  model, focal, psnrs, training_losses, valid_losses, iternums, valid_iternums = run(train_data =  TRAIN_DATA, 
      test_data = TEST_DATA, 
      channel = 2,
      near = NEAR,
      far = FAR,
      D = DEPTH,
      W = WIDTH_NETWORK,
      epochs = EPOCHS,
      L_embed = L_EMBED,
      ray_samples = RAY_SAMPLES,
      psnr_samples = PSNR_SAMPLES,
      see = SEE,
      to_save = True,
      limit = None,
      lr = LR)
  wandb.finish()
  




