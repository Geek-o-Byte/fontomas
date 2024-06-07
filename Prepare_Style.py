#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import os
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Initialize lists to store image data and labels
data = []
labels = []

font_folder = "font_images"

# Iterate through each font folder
for folder_name in os.listdir(font_folder):
    font_path = os.path.join(font_folder, folder_name)

    # Check if it's a directory
    if os.path.isdir(font_path):
        # Iterate through images in the font folder with tqdm progress bar
        for filename in tqdm(os.listdir(font_path), desc=f'Processing {folder_name}'):
            if filename.endswith('.png'):
                # Load the image
                image_path = os.path.join(font_path, filename)
                image = Image.open(image_path).convert("L")
                
                #print(image.shape)
                try:
                    # Convert the image to numpy array
                    resized_array = np.array(image)

                    # Normalize the image
                    normalized_image = resized_array / 127.5 - 1  # Assuming pixel values range from 0 to 255

                    # Append the normalized image data to the list
                    data.append(normalized_image)

                    # Extract label from filename (remove extension)
                    label = os.path.splitext(filename)[0]
                    labels.append(label)

                    # Print the shape of the saved image
                    #print(f"Shape of saved image: {normalized_image.shape}")
                except Exception as e:
                    #print(f"Error processing {filename}: {str(e)}")
                    continue

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)


# In[3]:


data.shape


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import math

timesteps = 600

# Pre-compute cosine schedule for beta
t_norm = np.linspace(0.0, 1.0, timesteps, endpoint=False)  # Avoid including timestep `timesteps-1`
beta = 0.01 + 0.01 * np.cos(2 * math.pi * t_norm)

# Alpha calculation (remains the same)
alpha = 1 - beta

# Pre-compute alpha bar
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)

# Pre-compute square root of alpha bar
sqrt_alpha_bar = np.sqrt(alpha_bar)

# Pre-compute one minus square root of alpha bar
one_minus_sqrt_alpha_bar = np.sqrt(1 - alpha_bar)

# this function will help us set the RNG key for Numpy
def set_key(key):
    np.random.seed(key)


# this function will add noise to the input as per the given timestamp
def forward_noise(key, x_0, t):
    set_key(key)
    noise = np.random.normal(size=x_0.shape)
    reshaped_sqrt_alpha_bar_t = np.reshape(np.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    noisy_image = reshaped_sqrt_alpha_bar_t * x_0 + reshaped_one_minus_sqrt_alpha_bar_t * noise
    return noisy_image, noise


# this function will be used to create sample timestamps between 0 & T
def generate_timestamp(key, num):
    set_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=timesteps, dtype=tf.int32)


# In[6]:


fig = plt.figure(figsize=(15, 30))

sample_mnist = data[599]

for index, i in enumerate([1, 10, 20, 30]):
    noisy_im, noise = forward_noise(0, np.expand_dims(sample_mnist, (0, -1)), np.array([i,]))
    plt.subplot(1, 4, index+1)
    plt.imshow(np.squeeze(np.squeeze(noisy_im, -1),0), cmap='gray')

plt.show()


# In[7]:


data.shape


# In[8]:


from tqdm import tqdm
import math

from tensorflow import keras, einsum
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from einops import rearrange
from einops.layers.tensorflow import Rearrange
from functools import partial
from inspect import isfunction


# In[9]:


# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[12]:


def preprocess(x, y):
    # Convert x to float32 and invert the colors of the image
    x = tf.subtract(255.0, tf.cast(x, tf.float32))
    # Resize and normalize
    x = tf.image.resize(x / 127.5 - 1, (32, 32))
    # Convert y to int32
    y = tf.cast(y, tf.int32)
    return x, y


# In[10]:


data[0].shape


# In[ ]:





# In[58]:


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# We will use this to convert timestamps to time encodings
class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        return emb


# small helper modules
class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)


class Residual(Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(x, training=training) + x


def Upsample(dim):
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')


def Downsample(dim):
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')


class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b
        return x


class PreNorm(Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def call(self, x, training=True):
        x = self.norm(x)
        return self.fn(x)


class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)


def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)


# building block modules
class Block(Layer):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = tfa.layers.GroupNormalization(groups, epsilon=1e-05)
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        x = self.proj(x)
        x = self.norm(x, training=training)

        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta

        x = self.act(x)
        return x


class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()

        self.mlp = Sequential([
            SiLU(),
            nn.Dense(units=dim_out * 2)
        ]) if exists(time_emb_dim) else None

        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = nn.Conv2D(filters=dim_out, kernel_size=1, strides=1) if dim != dim_out else Identity()

    def call(self, x, time_emb=None, training=True):
        gamma_beta = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)

        h = self.block1(x, gamma_beta=gamma_beta, training=training)
        h = self.block2(h, training=training)

        return h + self.res_conv(x)


class LinearAttention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.attend = nn.Softmax()
        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            LayerNorm(dim)
        ])

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)

        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)

        q = q * self.scale
        context = einsum('b h d n, b h e n -> b h d e', k, v)

        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
        out = self.to_out(out, training=training)

        return out


class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim_max = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
        sim_max = tf.cast(sim_max, tf.float32)
        sim = sim - sim_max
        attn = tf.nn.softmax(sim, axis=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=h, y=w)
        out = self.to_out(out, training=training)

        return out


class MLP(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.net = Sequential([
            Rearrange('... -> ... 1'),  # expand_dims(axis=-1)
            nn.Dense(units=hidden_dim),
            GELU(),
            LayerNorm(hidden_dim),
            nn.Dense(units=hidden_dim),
            GELU(),
            LayerNorm(hidden_dim),
            nn.Dense(units=hidden_dim),
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)


import tensorflow as tf

class StyleEncoder(Layer):
  def __init__(self, res, num_channels=1):
    super().__init__()

    self.block = Sequential([
      nn.Conv2D(32, (3, 3), activation='relu', padding='same'),
      nn.MaxPooling2D((2, 2)),
      nn.Conv2D(64, (3, 3), activation='relu', padding='same'),
      nn.MaxPooling2D((2, 2)),
      nn.Flatten(),
      nn.Dense(res * res * num_channels),
      SiLU(),
      nn.Reshape((res, res, num_channels))
    ])
    self.block.compile()

    #self.reduce_mean = nn.Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims=True))
  def call(self, x):
    #encoded = self.block(x)
    #encoded = self.reduce_mean(encoded)  # Get style representation
    return self.block(x)

class ClassConditioning(Layer):
    def __init__(self, res, num_channels=1):
        super().__init__()
        self.block = Sequential([
            nn.Dense(res * res * num_channels),
            SiLU(),
            nn.Reshape((res, res, num_channels))
        ])

        self.block.compile()

    def call(self, x):
        return self.block(x)


class Unet_conditional(Model):
    def __init__(self,
                 dim=64,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 resnet_block_groups=8,
                 learned_variance=False,
                 sinusoidal_cond_mlp=True,
                 num_classes=None,
                 class_embedder=None,
                 class_emb_dim=64,
                 in_res=64
                 ):
        super(Unet_conditional, self).__init__()

        # determine dimensions
        self.channels = channels
        self.in_res = in_res

        self.class_embeddings = nn.Embedding(num_classes, class_emb_dim) if class_embedder is None else class_embedder

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp

        if sinusoidal_cond_mlp:
            self.time_mlp = Sequential([
                SinusoidalPosEmb(dim),
                nn.Dense(units=time_dim),
                GELU(),
                nn.Dense(units=time_dim)
            ], name="time embeddings")
        else:
            self.time_mlp = MLP(time_dim)

        # layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        now_res = in_res

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                ClassConditioning(now_res),
                StyleEncoder(now_res),
                block_klass(dim_in + 1, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity()
            ])

            now_res //= 2 if not is_last else 1

        mid_dim = dims[-1]
        self.mid_class_conditioning = ClassConditioning(now_res)
        self.mid_style_conditioning = StyleEncoder(now_res)
        self.mid_block1 = block_klass(mid_dim + 1, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                ClassConditioning(now_res),
                StyleEncoder(now_res),
                block_klass((dim_out * 2) + 1, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity()
            ])

            now_res *= 2 if not is_last else 1

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = Sequential([
            block_klass(dim * 2, dim),
            nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
        ], name="output")

    def call(self, x, time=None, class_vector=None, style_vector = None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        class_vector = self.class_embeddings(class_vector)
        style_vector = np.random.rand(1, 24, 24, 1) if style_vector is None else style_vector

        h = []

        for class_conditioning, style_conditioning, block1, block2, attn, downsample in self.downs:
            cv = class_conditioning(class_vector)
            st  = style_conditioning(style_vector)
            #st = tf.math.reduce_mean(st, axis=-1)
            print(x.shape, cv.shape, st.shape)
            x = tf.concat([x, cv, st], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        cv = self.mid_class_conditioning(class_vector)
        st = self.mid_style_conditioning(style_vector)
        #st = tf.math.reduce_mean(st, axis=-1)
        x = tf.concat([x, cv, st], axis=-1)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for class_conditioning, style_conditioning, block1, block2, attn, upsample in self.ups:
            cv = class_conditioning(class_vector)
            st = style_conditioning(style_vector)
            #st = tf.math.reduce_mean(st, axis=-1)
            x = tf.concat([x, cv, st], axis=-1)
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = tf.concat([x, h.pop()], axis=-1)
        x = self.final_conv(x)
        return x

unet = Unet_conditional(
    num_classes=56,
    in_res=24,
    channels=1
)

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss


# In[52]:


test_images = np.ones([1, 24, 24, 1])
test_timestamps = generate_timestamp(0, 1)
test_class = np.array([3], dtype=np.int32)
k = unet(test_images, test_timestamps, test_class, np.random.rand(2, 24, 24, 1))
opt = keras.optimizers.Adam(learning_rate=1e-4)


# In[ ]:





# In[59]:


def train_step(batch, _class, _style_images):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    noised_image, noise = forward_noise(rng, batch, tf.cast(timestep_values, tf.int32))
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values, _class, _style_images)

        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value


# In[31]:


labels


# In[17]:


labels = labels.astype(np.int32)


# In[18]:


labels[5443]


# In[19]:


labels = labels - 1


# In[20]:


labels[0]


# In[21]:


data = np.expand_dims(data, -1)


# In[32]:


data.shape


# In[23]:


ckpt = tf.train.Checkpoint(unet=unet)
ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints/conditional_styled_diffusion/", max_to_keep=2)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


# In[33]:


import random

def ret_style_random(x, n):
    k = (x // 56) * 56
    subset = data[k : k+56]
    subset = subset.reshape(-1, 24, 24, 1)  # Reshape to (56, 24, 24, 1)
    subset_indices = random.sample(range(len(subset)), min(n, len(subset)))
    return subset[np.array(subset_indices)]


# In[48]:


ret_style_random(4, 4).shape


# In[60]:


BATCH_SIZE = 16
epochs = 6

for e in range(1, epochs+1):
    bar = tf.keras.utils.Progbar(len(data))
    losses = []
    
    for i in range(0, len(data), BATCH_SIZE):
        image_batch = data[i:i+BATCH_SIZE]
        label_batch = labels[i:i+BATCH_SIZE]
        
        # Run the training loop
        loss = train_step(image_batch, label_batch, ret_style_random(i, 12))
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    avg_loss = np.mean(losses)
    print(f"Average loss for epoch {e}/{epochs}: {avg_loss}")
    # Save the checkpoint
    ckpt_manager.save(checkpoint_number=e)


# In[28]:


def ddpm(x_t, pred_noise, t):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

    var = np.take(beta, t)
    z = np.random.normal(size=x_t.shape)

    return mean + (var ** .5) * z


# In[29]:


# ddpm
x = tf.random.normal((1,24,24,1))

img_list = []
img_list.append(np.squeeze(np.squeeze(x, 0), -1))

_class = 3

for i in tqdm(range(timesteps-1)):
    t = np.expand_dims(np.array(timesteps-i-1, np.int32), 0)
    pred_noise = unet(x, t, np.array([_class], dtype=np.int32))
    x = ddpm(x, pred_noise, t)
    img_list.append(np.squeeze(np.squeeze(x, 0), -1))

    if i % 20==0:

        plt.imshow(x[0], cmap='gray')
        plt.show()

plt.imshow(x[0], cmap='gray')
plt.show()


# In[30]:


def ddim(x_t, pred_noise, t, sigma_t):
    alpha_t_bar = np.take(alpha_bar, t)
    alpha_t_minus_one = np.take(alpha, t-1)

    pred = (x_t - ((1 - alpha_t_bar) ** 0.5) * pred_noise)/ (alpha_t_bar ** 0.5)
    pred = (alpha_t_minus_one ** 0.5) * pred

    pred = pred + ((1 - alpha_t_minus_one - (sigma_t ** 2)) ** 0.5) * pred_noise
    eps_t = np.random.normal(size=x_t.shape)
    pred = pred+(sigma_t * eps_t)

    return pred


# In[82]:


from random import randint


for cl_name in range(0, 56):
    # Define number of inference loops to run
    _class = cl_name
    
    inference_timesteps = 5
    
    # Create a range of inference steps that the output should be sampled at
    inference_range = range(0, timesteps, timesteps // inference_timesteps)
    
    x = tf.random.normal((1,24,24,1))
    img_list = []
    img_list.append(np.squeeze(np.squeeze(x, 0),-1))
    
    print(np.array([_class], dtype=np.int32))
    
    # Iterate over inference_timesteps
    for index, i in tqdm(enumerate(reversed(range(inference_timesteps))), total=inference_timesteps):
        t = np.expand_dims(inference_range[i], 0)
    
        pred_noise = unet(x, t, np.array([_class], dtype=np.int32))
    
        x = ddim(x, pred_noise, t, 0)
        img_list.append(np.squeeze(np.squeeze(x, 0),-1))   
        
        # if index % 1 == 0:
        #     plt.imshow(np.array(np.clip((np.squeeze(np.squeeze(x, 0),-1) + 1) * 127.5, 0, 255), np.uint32), cmap="gray")
        #     plt.show()
    
    plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255), np.uint32), cmap="gray")
    
    plt.savefig(f'foos/foo_{cl_name}_{randint(0, 100000)}.png')
    
    plt.show()


# In[ ]:





# In[ ]:




