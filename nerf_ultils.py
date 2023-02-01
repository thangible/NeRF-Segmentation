
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from model import  render_rays_segment, get_rays
from dataset import get_transform_data
import imageio

def visualize(ground_true, predicted,
              epoch: int = None, 
              iternums: list = None, 
              valid_iternums: list = None,
              psnrs: list = None,
              to_save: bool = False, 
              save_dir: str = './figs/' , 
              training_losses = None,
              valid_losses = None):
  plt.figure(figsize=(20,10))
  plt.subplot(141)
  plt.imshow(ground_true)
  plt.subplot(142)
  plt.imshow(predicted) #
  plt.title(f'Iteration: {epoch}')
  if psnrs:
    plt.subplot(143)
    plt.plot(iternums, psnrs)
    plt.title('PSNR')
  if training_losses and valid_losses:
    plt.subplot(144)
    plt.plot(iternums, training_losses, label = 'training loss')
    plt.plot(valid_iternums, valid_losses, label = 'valid loss')
    plt.legend()
    plt.title('loss')
  if to_save:
    pic_name = 'image_at_epoch_{:04d}.png'.format(epoch)
    path = os.path.join(save_dir, pic_name)
    if not(os.path.exists(save_dir)):
      os.mkdir(save_dir)
    plt.savefig(path)
  plt.show()  
  return plt

def get_avg_PSNR(model, test_data, sample_num , near, far, ray_samples, L_embed):
  import random
  test_imgs, test_transform = test_data
  length, height, width = test_imgs.shape[:3]
  test_poses, focal, _ = get_transform_data(test_transform, width)
  psnrs = []
  for i in range(sample_num):
      n = random.randint(0, length -1)
      test_pose = test_poses[n]
      test_img = test_imgs[n]
      rays_o, rays_d = get_rays(height, width, focal, test_pose)
      rays_o = tf.cast(rays_o, tf.float32) #edit
      rays_d = tf.cast(rays_d, tf.float32) #edit
      predicted, _, _, _ = render_rays_segment(model, rays_o, rays_d, near= near, far= far, N_samples=ray_samples, L_embed = L_embed)
      loss = tf.reduce_mean(tf.square(predicted - test_img))
      psnr = -10. * tf.math.log(loss) / tf.math.log(10.)
      psnrs.append(psnr.numpy())
  return np.mean(psnrs)

def get_2d_from_pose(ground_truth,model, pose, height, width, focal, near, far, ray_samples, L_embed, epochs, psnrs):
  rays_o, rays_d = get_rays(height, width, focal, pose)
  rays_o = tf.cast(rays_o, tf.float32) #edit
  rays_d = tf.cast(rays_d, tf.float32) #edit
  predicted, _, _, _ = render_rays_segment(model, rays_o, rays_d, near= near, far= far, N_samples= ray_samples, L_embed = L_embed)
  pic = visualize(ground_truth, predicted, epoch = epochs, iternums = None, psnrs= psnrs)
  return predicted, pic




def pose_spherical(theta, phi, radius):
  trans_t = lambda t : tf.convert_to_tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=tf.float32)

  rot_phi = lambda phi : tf.convert_to_tensor([
      [1,0,0,0],
      [0,tf.cos(phi),-tf.sin(phi),0],
      [0,tf.sin(phi), tf.cos(phi),0],
      [0,0,0,1],
  ], dtype=tf.float32)

  rot_theta = lambda th : tf.convert_to_tensor([
      [tf.cos(th),0,-tf.sin(th),0],
      [0,1,0,0],
      [tf.sin(th),0, tf.cos(th),0],
      [0,0,0,1],
  ], dtype=tf.float32)

  c2w = trans_t(radius)
  c2w = rot_phi(phi/180.*np.pi) @ c2w
  c2w = rot_theta(theta/180.*np.pi) @ c2w
  c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
  return c2w

def get_gif(model, near, far, ray_samples, H, W, focal, theta_interval = 60, phi = -30., radius = 2.):
  frames = []
  for th in tqdm(np.linspace(0., 360., theta_interval, endpoint=False)):
      c2w = pose_spherical(th, phi, radius)
      rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
      rays_o = tf.cast(rays_o, tf.float32) #edit
      rays_d = tf.cast(rays_d, tf.float32) #edit
      pic, depth, acc, pts = render_rays_segment(model, rays_o, rays_d, near=near, far=far, N_samples=ray_samples)
      pic = pic != 0.0
      pic = tf.cast(pic, tf.int8)
      frames.append(~pic)
  
  f = 'video.mp4'
  fps = 2 *360 // theta_interval
  kargs = { 'macro_block_size': None }
  print(fps)
  imageio.mimwrite(f, frames, fps=fps, quality=7, **kargs)
  return f