
import os

import tensorflow as tf
import glob
import numpy as np
import time
import pandas as pd
import cv2

from ConvNorm import ConvNorm

#%%
PATH = 'dataset'

#%%
BUFFER_SIZE = 50
BATCH_SIZE = 4
PATCH_SIZE = 256


def load_train(inf, trf, htf):
    C = np.array([[17, 141, 215],
                  [225, 227, 155],
                  [127, 173, 123],
                  [185, 122, 87],
                  [230, 200, 181],
                  [150, 150, 150],
                  [193, 190, 175]])
    C = np.reshape(C, (1,1,C.shape[0],3))
    
    input_image = tf.io.decode_png(tf.io.read_file(inf))
    real_image = tf.io.decode_png(tf.io.read_file(trf))
    height_image = tf.io.decode_png(tf.io.read_file(htf), dtype=tf.uint16)
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    height_image = tf.cast(height_image, tf.float32)
    
    orig_image = tf.image.resize(real_image, [PATCH_SIZE, PATCH_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True) / 127.5 - 1
    orig_image.set_shape([None, None, 3])
    
    temp = tf.image.random_crop(tf.stack([input_image, real_image, tf.concat([height_image,tf.zeros_like(height_image),tf.zeros_like(height_image)], axis=2)], axis=0), size=[3, PATCH_SIZE, PATCH_SIZE, 3])
    input_image = temp[0]
    real_image = temp[1]
    height_image = tf.expand_dims(temp[2,:,:,0], -1)
    
    #patch2 = tf.image.random_crop(real_image, size=[PATCH_SIZE, PATCH_SIZE,3]) / 127.5 - 1
    
    input_image = tf.one_hot(tf.argmin(tf.norm(tf.expand_dims(input_image, -2)-C, axis=3), 2), C.shape[2], dtype=tf.float32)
    real_image = real_image / 127.5 - 1
    height_image = height_image / 32767.5 - 1
    
    return real_image, input_image, height_image, orig_image


#%%
files = glob.glob(os.path.join(PATH, '*_i2.png'))
tfiles = [f.replace('_i2.png', '_t.png') for f in files]
hfiles = [f.replace('_i2.png', '_h.png') for f in files]

train_dataset = tf.data.Dataset.from_tensor_slices((files, tfiles, hfiles))
train_dataset = train_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


#%%
def conv(x, channels, kernel=3, stride=1, pad=0, pad_type='symmetric', use_bias=True):
    if kernel>1:
        p = (kernel-1)//2
        x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')
    x = tf.keras.layers.Conv2D(channels, kernel, strides=stride, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=use_bias, bias_initializer=tf.initializers.constant(0.0))(x)
    return x
    

def resblock(x_init, channels, stride=1):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)
    
    x_init = tf.nn.leaky_relu(x_init, 0.2)
    #x = conv(x_init, channel_middle, stride=stride)
    x = ConvNorm(channel_middle, strides=stride)(x_init)
    
    x = tf.nn.leaky_relu(x, 0.2)
    #x = conv(x, channel_middle)
    x = ConvNorm(channel_middle)(x)
    
    x = tf.nn.leaky_relu(x, 0.2)
    #x = conv(x, channels)
    x = ConvNorm(channels)(x)
    
    if channel_in != channels or stride>1:
        #x_init = conv(x_init, channels, kernel=1, stride=stride)
        x_init = ConvNorm(channels, kernel_size=1, strides=stride)(x_init)

    return x + x_init
    

#%%
OUTPUT_CHANNELS = 128

def Encoder(in_channels=3, out_channels=512, blocks=4):
    x_init = tf.keras.layers.Input(shape=[None,None,in_channels])
    #batch_size = tf.shape(x)[0]
    
    channels = 64
    
    x = conv(x_init, channels, kernel=5, stride=2)
    
    for i in range(blocks):
        x = resblock(x, channels*2**i, stride=2)
    
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv(x, channels=out_channels, kernel=1)
    x = tf.math.reduce_mean(x, axis=(1,2))
    
    return tf.keras.Model(inputs=x_init, outputs=x)


temp = []
for i in tf.data.experimental.get_structure(train_dataset):
    temp.append(i.shape[3])
Encoders = [Encoder(in_channels=i, out_channels=OUTPUT_CHANNELS, blocks=5) for i in temp]
#Encoders += [Encoders[0]]
combinations = [(0,1),(0,2),(0,3),(1,0),(1,2),(2,0),(2,1),(3,0)]

#%%
class MemoryMoCo(tf.keras.layers.Layer):
    def __init__(self, n_views, q_size=1024, T=0.07, combinations=[]):
        super(MemoryMoCo, self).__init__()
        #self.input_size = input_size
        self.n_views = n_views
        self.q_size = q_size
        self.T = T
        
        if len(combinations)==0:
            self.combinations = []
            for i in range(self.n_views):
                for j in np.arange(self.n_views):
                    if i!=j:
                        self.combinations.append((i,j))
        else:
            self.combinations = combinations
        
        
    def build(self, input_shape):
        self.input_size = input_shape[0][1]
        
        
    def call(self, X, Y, Mem):
        batch = X[0].shape[0]
        
        nMem = [0]*len(Mem)
        for i in range(len(Mem)):
            if Mem[i]==None:
                std = 1. / np.sqrt(self.input_size / 3)
                nMem[i] = np.random.normal(size=(1, self.q_size, self.input_size)).astype('float32')*(2*std)-std
            else:
                nMem[i] = Mem[i]
        
        out = []
        Xn = [tf.linalg.norm(i, axis=1, keepdims=True) for i in X]
        for n,a in enumerate(self.combinations):
            i,j = a
            
            M = tf.concat((tf.expand_dims(X[j], 1), tf.repeat(nMem[j], batch, axis=0)), axis=1, name=f'concat_{i}{j}')
            temp = tf.squeeze(tf.matmul(M, tf.expand_dims(X[i], -1)))
            #N = tf.linalg.norm(X[i], axis=1, keepdims=True)*tf.linalg.norm(X[j], axis=1, keepdims=True)
            N = Xn[i]*Xn[j]
            temp = tf.math.exp(temp / N / self.T)
            out.append(temp)
            
        for i in range(len(nMem)):
            nMem[i] = tf.concat((nMem[i][:, batch:, :], tf.expand_dims(Y[i], 0)), axis=1, name=f'update_{i}')
        
        return out, nMem

Q = 2048
Memory = MemoryMoCo(len(Encoders), q_size=Q, combinations=combinations)


def NCEloss(x, q_size):
    q_size = tf.cast(q_size, tf.float32)
    eps = 1e-7
    bsz = tf.cast(tf.shape(x)[0], tf.float32)

    Pn =  q_size / (q_size+1)

    P_pos = x[:, 0]
    log_D1 = tf.math.log(P_pos / (P_pos + Pn + eps))

    P_neg = x[:, 1:]
    log_D0 = tf.math.log(tf.ones_like(P_neg) * Pn / (P_neg + Pn + eps))
    
    loss = - (tf.math.reduce_sum(log_D1) + tf.math.reduce_sum(log_D0)) / bsz
    loss = tf.clip_by_value(loss, -1e10, 1e10)
    return loss




#%%

optimizer = tf.keras.optimizers.Adam(1e-4, 0.9, clipnorm=100)


checkpoint_dir = r'ckpt'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.listed = Encoders

manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5, keep_checkpoint_every_n_hours=2)
status = checkpoint.restore(manager.latest_checkpoint)


Encoders_ema = [tf.keras.models.clone_model(i) for i in Encoders]
for i in range(len(Encoders_ema)):
    Encoders_ema[i].trainable=False


#%%
@tf.function
def train_step(inputs, Mem):
    #Q = 2048
    batch = inputs[0].shape[0]
    with tf.GradientTape() as grad_tape:
        
        x = []
        y = []
        for i in range(len(inputs)):
            x.append(tf.reshape(Encoders[i](inputs[i]), (batch, -1)))
            y.append(tf.stop_gradient(tf.reshape(Encoders_ema[i](inputs[i]), (batch, -1))))
        
        loss, Mem = Memory(x, y, Mem)
        loss = [NCEloss(i, Q) for i in loss]
        if tf.math.reduce_any([(tf.math.is_nan(i) or tf.math.is_inf(i)) for i in loss]):
            tf.print(loss)
        Loss = tf.math.reduce_sum(loss)
        
    V = []
    for i in set(Encoders):
        V += i.trainable_variables
    grad = grad_tape.gradient(Loss, V)
    if tf.math.is_inf(tf.linalg.global_norm(grad)):
        grad = [tf.clip_by_value(i, -1e16, 1e16) for i in grad]
    optimizer.apply_gradients(zip(grad, V))
    
    m = 0.99
    for i in range(len(Encoders)):
        for j in range(len(Encoders[i].variables)):
            Encoders_ema[i].variables[j] = m*Encoders_ema[i].variables[j] + (1-m)*Encoders[i].variables[j]
    #print('EMA')
    
    return loss, Mem, x[0]


def fit(train_ds, epochs):
    datpath = 'data'
    Mem0 = [None]*len(Encoders)
    
    L = []
    X = []
    IM = []
    for epoch in range(epochs):
        start = time.time()
        
        # Train
        loss = []
        x = []
        im = []
        for inputs in train_ds:
            l, Mem, emb = train_step(inputs, Mem0)
            
            im.append(np.array((inputs[0]+1)*127.5).astype('uint8'))
            l = np.array(l)
            Mem0 = [np.array(m) for m in Mem]
            loss.append(np.array(l))
            x.append(np.array(emb))
            
        
        im = np.vstack(im)
        pd.DataFrame(np.array(loss)).to_csv(os.path.join(datpath, f'L{epoch:02d}.csv'), index=None, header=None)
        pd.DataFrame(np.vstack(x)).to_csv(os.path.join(datpath, f'X{epoch:02d}.csv'), index=None, header=None)
        
        imout = np.zeros((100*256, 50*256, 3), dtype='uint8')
        for i in range(100):
            for j in range(50):
                imout[256*i:256*(i+1), 256*j:256*(j+1), :] = im[i*50+j,:,:,::-1]
        cv2.imwrite(os.path.join(datpath, f'I{epoch:02d}.png'), imout)
        
        L = [np.array(loss)]
        X = [np.vstack(x)]
        IM = [im]
        
        if (epoch + 1) % 1 == 0:
            manager.save()
        
        print('Epoch {} took {} min'.format(epoch, np.round((time.time()-start)/60, 2)))
        print(np.median(L[-1], axis=0))
        if np.any(np.isnan(L[-1])):
            break
    return L, X, IM

#%%
EPOCHS = 20
L, X, IM = fit(train_dataset, EPOCHS)

#%%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from umap import UMAP

U0 = UMAP(n_neighbors=100).fit(X[-1])
U = U0.embedding_

plt.scatter(U[:,0],U[:,1],c='k')
ax = plt.gca()
for i in np.random.choice(IM[-1].shape[0], 800, replace=False):
    im = cv2.resize(IM[-1][i,:,:,:], (150,150))
    imagebox = OffsetImage(im, zoom=0.25)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, U[i,:2], xybox=(0, 0), xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)

