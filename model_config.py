import numpy as np
import tensorflow as tf

def embed_fn(x, L_embed = 6): # embed 1d to 2x L_embed dimensions #positional encoding
  rets = [x]
  for i in range(L_embed):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)

def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy') # i (H,W) , j (W,H)
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1) #
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) # org of vectors # Shape (H,W,3)
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d)) #direction of ray  # Shape (H,W,3)
    return rays_o, rays_d #

def init_model(D=8*2, W=64, channel = 4, L_embed = 6):
    relu = tf.keras.layers.ReLU()    
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act) 

    inputs = tf.keras.Input(shape=(3 + 3*2*L_embed)) 
    outputs = inputs
    for i in range(D):
        outputs = dense()(outputs)
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs], -1)
    outputs = dense(channel, act=None)(outputs) 
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
  
def render_rays_segment(network_fn, rays_o, rays_d, near, far, N_samples, rand=False, channel = 4, L_embed = 6):
    def batchify(fn, chunk=1024*32):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    # Compute 3D query points
    z_vals = tf.linspace(near, far, N_samples) 
    if rand:
      z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # shape = H, W, z_vals.num, 3
    
    # Run network
    pts_flat = tf.reshape(pts, [-1,3]) # SHape = H*W*y_val.num , 3 (bunch of points) 
    pts_flat = embed_fn(pts_flat, L_embed)
    raw = batchify(network_fn)(pts_flat) #shape = batches, 4
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [channel]) # shape = H, W, z_vals.num, 4
    
    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,channel-1]) #H, W, z_vals.num, 1
    channel_map_raw = tf.math.sigmoid(raw[...,:channel-1]) #H, W, z_vals.num, 3
    
    # Do volume rendering    
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1) 
    alpha = 1.-tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    depth_map = tf.reduce_sum(weights * z_vals, -1) 
    acc_map = tf.reduce_sum(weights, -1)
    
    if channel != 1:
      gray_map = tf.reduce_max(sigma_a, axis = -1)
    else:
      channel_map = tf.reduce_sum(weights[...,None] * channel_map_raw, -2) 
      gray_map = tf.reduce_mean(channel_map, -1)
    return gray_map, depth_map, acc_map, pts