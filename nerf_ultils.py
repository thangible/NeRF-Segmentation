
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import imageio
import tqdm

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

def spherical_to_matrix(theta, phi, radius):
    # Convert from spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    # Create the translation vector
    t = np.array([x, y, z, 1])
    # Create the rotation matrix
    R_x = np.array([[1,         0,                  0,                  0],
                    [0,         np.cos(theta), -np.sin(theta),     0],
                    [0,         np.sin(theta),  np.cos(theta),     0],
                    [0,         0,                  0,                  1]
                   ])
    R_y = np.array([[np.cos(phi),    0,      np.sin(phi), 0],
                    [0,                 1,      0,                 0],
                    [-np.sin(phi), 0,      np.cos(phi), 0],
                    [0,                 0,      0,                 1]
                   ])
    R = np.matmul(R_y, R_x)
    # Create the transformation matrix
    T = np.array([[R[0][0], R[0][1], R[0][2], t[0]],
                  [R[1][0], R[1][1], R[1][2], t[1]],
                  [R[2][0], R[2][1], R[2][2], t[2]],
                  [0,             0,             0,             t[3]]
                 ])
    return T

def matrix_to_spherical(matrix):
    # Extract the rotation matrix from the transformation matrix
    R = matrix[0:3, 0:3]

    # Extract the translation vector from the transformation matrix
    t = matrix[0:3, 3]
    
    # Calculate the radius
    radius = np.linalg.norm(t)
    
    # Calculate theta
    theta = np.arccos(R[2][2] / radius)
    
    # Calculate phi
    phi = np.arctan2(R[1][2], R[0][2])
    
    return theta, phi, radius

def image_grid(x, size=6):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) 
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]