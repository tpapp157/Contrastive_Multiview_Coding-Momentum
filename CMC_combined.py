
import os

import tensorflow as tf
import glob
import numpy as np
import time
import pandas as pd
import cv2

from ConvNorm import ConvNorm


#%%
PATH = r'dataset'

#%%
BUFFER_SIZE = 50
BATCH_SIZE = 5
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



def load_test(imf):
    im = tf.io.decode_png(tf.io.read_file(imf))
    
    H = tf.shape(im)[0]
    W = tf.shape(im)[1]
    h = (H-PATCH_SIZE)//2
    w = (W-PATCH_SIZE)//2
    
    im = im[h:h+PATCH_SIZE, w:w+PATCH_SIZE, :]
    im = tf.cast(im, tf.float32) / 127.5 - 1
    return im


#%%
files = glob.glob(os.path.join(PATH, '*_i2.png'))
tfiles = [f.replace('_i2.png', '_t.png') for f in files]
hfiles = [f.replace('_i2.png', '_h.png') for f in files]

train_dataset = tf.data.Dataset.from_tensor_slices((files, tfiles, hfiles)).shuffle(len(files)).map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(tfiles).map(load_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)


#%%
def conv(x, channels, kernel=3, stride=1, pad=0, pad_type='symmetric', use_bias=True):
    if kernel>1:
        p = (kernel-1)//2
        x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')
    x = tf.keras.layers.Conv2D(channels, kernel, strides=stride, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=use_bias, bias_initializer=tf.initializers.constant(0.0))(x)
    return x
    

def resblock(x_init, stride=1):
    channel_in = x_init.get_shape().as_list()[-1]
    n = 1
    if stride>1:
        n = 2
    
    x_init = tf.nn.leaky_relu(x_init, 0.2)
    #x = conv(x_init, channel_middle, stride=stride)
    x = ConvNorm(channel_in//2, kernel_size=1)(x_init)
    
    x = tf.nn.leaky_relu(x, 0.2)
    #x = conv(x, channel_middle)
    x = ConvNorm(channel_in//2, strides=stride)(x)
    
    x = tf.nn.leaky_relu(x, 0.2)
    #x = conv(x, channels)
    x = ConvNorm(channel_in*n, kernel_size=1)(x)
    
    if stride>1:
        #x_init = conv(x_init, channels, kernel=1, stride=stride)
        x_init = ConvNorm(channel_in*n, kernel_size=1, strides=stride)(x_init)
    
    return x + x_init


def regnet_block(x_init, channels, g, stride=1):
    channel_in = x_init.get_shape().as_list()[-1]
    
    x_init = tf.nn.leaky_relu(x_init, 0.2)
    #x_init = tf.nn.relu(x_init)
    x = ConvNorm(channels, kernel_size=1)(x_init)
    
    x = tf.nn.leaky_relu(x, 0.2)
    #x = tf.nn.relu(x)
    x = ConvNorm(channels, kernel_size=3, group_size=g, strides=stride)(x)
    
    x = tf.nn.leaky_relu(x, 0.2)
    #x = tf.nn.relu(x)
    x = ConvNorm(channels, kernel_size=1)(x)
    
    if channel_in!=channels:
        x_init = ConvNorm(channels, kernel_size=1, strides=stride)(x_init)
    
    return x + x_init
    
    

def regnetx(x, D, W, G):
    assert len(D)==len(W)
    
    for i,d in enumerate(D):
        for j in range(d):
            if j==0 and i>0:
                x = regnet_block(x, W[i], G, stride=2)
            else:
                x = regnet_block(x, W[i], G)
    
    return x




#%%
OUTPUT_CHANNELS = 64
START_CHANNELS = 32

def FeatureExtractor(in_channels, blocks=5):
    x_init = tf.keras.layers.Input(shape=[None, None, in_channels])
    x = x_init
    
    for i in range(blocks):
        x = resblock(x, stride=2)
        x = resblock(x)
    
    return tf.keras.Model(inputs=x_init, outputs=x)


def FeatureExtractor_regnet(D, W, G):
    x_init = tf.keras.layers.Input(shape=[None, None, W[0]])
    x = regnetx(x_init, D, W, G)
    
    return tf.keras.Model(inputs=x_init, outputs=x)


def Classifier(in_channels, out_channels=128):
    x_init = tf.keras.layers.Input(shape=[None, None, in_channels])
    x = tf.math.reduce_mean(x_init, axis=(1,2))
    
    x = tf.nn.leaky_relu(x, 0.2)
    x = tf.keras.layers.Dense(units=in_channels)(x)
    
    x = tf.nn.leaky_relu(x, 0.2)
    x = tf.keras.layers.Dense(units=out_channels)(x)
    
    x = tf.nn.leaky_relu(x, 0.2)
    x = tf.keras.layers.Dense(units=out_channels)(x)
    
    return tf.keras.Model(inputs=x_init, outputs=x)


def Att_Classifier(in_channels, out_channels=128, h=3, n=64):
    x_init = tf.keras.layers.Input(shape=[None, None, in_channels])
    b = tf.shape(x_init)[0]
    
    x = tf.nn.leaky_relu(x_init, 0.2)
    x = ConvNorm(in_channels, kernel_size=1)(x)
    
    q = tf.keras.layers.Dense(units=h*n)(tf.nn.leaky_relu(tf.math.reduce_mean(x, axis=(1,2)), 0.2))
    q = tf.nn.relu(tf.reshape(q, (-1, h, n)))
    x = tf.nn.leaky_relu(x, 0.2)
    k = tf.reshape(conv(x, n, 1), (b, -1, n))
    v = tf.reshape(conv(x, n, 1), (b, -1, n))
    
    k = tf.nn.softmax(tf.matmul(q, tf.transpose(k, (0,2,1))) / tf.math.sqrt(tf.cast(n, tf.float32)))
    x = tf.reshape(tf.matmul(k, v), (-1, h*n))
    x = tf.keras.layers.Dense(units=out_channels)(x)
    
    return tf.keras.Model(inputs=x_init, outputs=x)


def Init_Layer(in_channels=3, channels=64):
    x_init = tf.keras.layers.Input(shape=[None, None, in_channels])
    x = conv(x_init, channels, kernel=5, stride=2)
    return tf.keras.Model(inputs=x_init, outputs=x)



Encoder = FeatureExtractor(START_CHANNELS, blocks=5)

# =============================================================================
# #D = [2, 5, 14, 2]
# #W = [80, 240, 560, 1360]
# #G = 40
# #D = [2, 6, 15, 2]
# #W = [96, 192, 432, 1008]
# #G = 48
# D = [2, 4, 10, 2]
# W = [72, 168, 408, 912]
# G = 24
# Encoder = FeatureExtractor_regnet(D, W, G)
# =============================================================================

FE_outchannels = Encoder.get_layer(index=-1).output_shape[0][-1]
FE_inchannels = Encoder.get_layer(index=0).output_shape[0][-1]

Init_Layers = [Init_Layer(in_channels=i.shape[3], channels=FE_inchannels) for i in tf.data.experimental.get_structure(train_dataset)[:3]]
Init_Layers += [Init_Layers[0]]

Classifiers = [Classifier(in_channels=FE_outchannels, out_channels=OUTPUT_CHANNELS) for _ in range(len(tf.data.experimental.get_structure(train_dataset)))]
#Classifiers = [Att_Classifier(in_channels=FE_outchannels, out_channels=OUTPUT_CHANNELS, n=64) for _ in range(len(tf.data.experimental.get_structure(train_dataset)))]

combinations = [(0,1),(0,2),(0,3),
                (1,0),(1,2),
                (2,0),(2,1),
                (3,0)]

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
        
        out0 = []
        Xn = [tf.linalg.norm(i, axis=1, keepdims=True) for i in X]
        Mn = [tf.repeat(tf.linalg.norm(i, axis=2), batch, axis=0) for i in nMem]
        for n,a in enumerate(self.combinations):
            i,j = a
            
            M = tf.concat((tf.expand_dims(X[j], 1), tf.repeat(nMem[j], batch, axis=0)), axis=1, name=f'concat_{i}{j}')
            temp = tf.squeeze(tf.matmul(M, tf.expand_dims(X[i], -1)))
            N = Xn[i] * tf.concat((Xn[j], Mn[j]), axis=1)
            temp = tf.math.exp(temp / N / self.T)
            out0.append(temp)
        
        out1 = []
        for n,a in enumerate(self.combinations):
            i,j = a
            M = []
            for k in np.arange(batch):
                M.append(tf.expand_dims(tf.concat((X[j][k:,:], X[j][:k,:]), axis=0), 0))
            M = tf.concat(M, axis=0)
            temp = tf.squeeze(tf.matmul(M, tf.expand_dims(X[i], -1)))
            N = Xn[i] * tf.linalg.norm(M, axis=2)
            temp = tf.math.exp(temp / N / self.T)
            out1.append(temp)
        
        
        for i in range(len(nMem)):
            nMem[i] = tf.concat((nMem[i][:, batch:, :], tf.expand_dims(Y[i], 0)), axis=1, name=f'update_{i}')
        
        return out0, out1, nMem


Q = 10*BATCH_SIZE
Memory = MemoryMoCo(len(Classifiers), q_size=Q, combinations=combinations)



def NCEloss(x, q_size):
    q_size = tf.cast(q_size, tf.float32)
    eps = 1e-7
    bsz = tf.cast(tf.shape(x)[0], tf.float32)
    
    Pn =  q_size / (q_size+1)
    
    P_pos = x[:, 0]
    log_D1 = tf.math.log(P_pos / (P_pos + Pn + eps) + eps)
    
    P_neg = x[:, 1:]
    log_D0 = tf.math.log(tf.ones_like(P_neg) * Pn / (P_neg + Pn + eps) + eps)
    
    loss = - (tf.math.reduce_sum(log_D1) + tf.math.reduce_sum(log_D0)) / bsz
    loss = tf.clip_by_value(loss, -1e10, 1e10)
    return loss




#%%
optimizer = tf.keras.optimizers.Adam(1e-4, 0.9, clipnorm=100)

checkpoint_dir = r'ckpt'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, Encoder=Encoder)
checkpoint.listed = Init_Layers
checkpoint.listed = Classifiers

manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5, keep_checkpoint_every_n_hours=2)
status = checkpoint.restore(manager.latest_checkpoint)


Encoder_ema = tf.keras.models.clone_model(Encoder)
# =============================================================================
# Encoder_ema = FeatureExtractor_regnet(D, W, G)
# for i,_ in enumerate(Encoder_ema.variables):
#     Encoder_ema.variables[i].assign(Encoder.variables[i].value())
# =============================================================================
Encoder_ema.trainable = False

Init_Layers_ema = [tf.keras.models.clone_model(i) for i in Init_Layers]
Classifiers_ema = [tf.keras.models.clone_model(i) for i in Classifiers]
for i in range(len(Classifiers_ema)):
    Classifiers_ema[i].trainable=False
    Init_Layers_ema[i].trainable=False


#%%
def test_model(in_channels=3):
    x_init = tf.keras.layers.Input(shape=[None, None, in_channels])
    x = Init_Layers[0](x_init)
    x = Encoder(x)
    return tf.keras.Model(inputs=x_init, outputs=x)
TestModel = test_model(in_channels=Init_Layers[0].get_layer(index=0).output_shape[0][-1])


#%%
@tf.function
def train_step(inputs, Mem):
    batch = inputs[0].shape[0]
    with tf.GradientTape() as grad_tape:
        
        x = [Init_Layers[i](inputs[i]) for i in range(len(inputs))]
        
        x0 = [Encoder(i) for i in x]
        #x0 = [tf.nn.avg_pool(i, 4, 4, padding='VALID') for i in x0]
        y = [tf.stop_gradient(Encoder_ema(i)) for i in x]
        
        y = [tf.stop_gradient(Classifiers_ema[i](Encoder_ema(x[i]))) for i in range(len(x))]
        x = [Classifiers[i](x0[i]) for i in range(len(x0))]
        
        l_inter, l_intra, Mem = Memory(x, y, Mem)
        loss = [NCEloss(i, Q) for i in l_inter] + [NCEloss(i, batch) for i in l_intra]
        if tf.math.reduce_any([(tf.math.is_nan(i) or tf.math.is_inf(i)) for i in loss]):
            tf.print(loss)
        Loss = tf.math.reduce_sum(loss)
        
    V = Encoder.trainable_variables
    for i in set(Init_Layers + Classifiers):
        V += i.trainable_variables
    grad = grad_tape.gradient(Loss, V)
    if tf.math.is_inf(tf.linalg.global_norm(grad)):
        grad = [tf.clip_by_value(i, -1e16, 1e16) for i in grad]
    optimizer.apply_gradients(zip(grad, V))
    
    m = 0.99
    for j in range(len(Encoder.variables)):
        Encoder_ema.variables[j] = m*Encoder_ema.variables[j] + (1-m)*Encoder.variables[j]
        
    for i in range(len(Init_Layers)):
        for j in range(len(Init_Layers[i].variables)):
            Init_Layers_ema[i].variables[j] = m*Init_Layers_ema[i].variables[j] + (1-m)*Init_Layers[i].variables[j]
            
    for i in range(len(Classifiers)):
        for j in range(len(Classifiers[i].variables)):
            Classifiers_ema[i].variables[j] = m*Classifiers_ema[i].variables[j] + (1-m)*Classifiers[i].variables[j]
    
    return loss, Mem, x0[0]


def fit(train_ds, epochs):
    datpath = 'data'
    if not os.path.isdir(datpath):
        os.makedirs(datpath)
    
    Mem0 = [None]*len(Classifiers)
    
    for epoch in range(epochs):
        start = time.time()
        
        # Train
        loss = []
        x = []
        im = []
        for inputs in train_ds:
            im.append(np.array((inputs[0]+1)*127.5).astype('uint8'))
            l, Mem, emb = train_step(inputs, Mem0)
            
            l = np.array(l)
            Mem0 = [np.array(m) for m in Mem]
            loss.append(np.array(l))
            x.append(np.mean(np.array(emb), axis=(1,2)))
        
        loss = np.array(loss)
        pd.DataFrame(loss).to_csv(os.path.join(datpath, f'L{epoch:02d}.csv'), index=None, header=None)
        print('Epoch {} took {} min'.format(epoch, np.round((time.time()-start)/60, 2)))
        print(np.median(loss, axis=0))
        
        pd.DataFrame(np.vstack(x)).to_csv(os.path.join(datpath, f'Xt{epoch:02d}.csv'), index=None, header=None)
        im = np.vstack(im)
        imout = np.zeros((100*256, 50*256, 3), dtype='uint8')
        for i in range(100):
            for j in range(50):
                imout[256*i:256*(i+1), 256*j:256*(j+1), :] = im[i*50+j,:,:,::-1]
        cv2.imwrite(os.path.join(datpath, f'It{epoch:02d}.png'), imout)
        
        if np.mod(epoch+1, 2)==0 or epoch==epochs-1:
            x = []
            im = []
            for inputs in test_dataset:
                im.append(np.array((inputs+1)*127.5).astype('uint8'))
                emb = TestModel(inputs)
                
                x.append(np.mean(np.array(emb), axis=(1,2)))
                #emb = tf.nn.avg_pool(emb, 4, 4, padding='VALID')
                #x.append(np.reshape(np.array(emb), (BATCH_SIZE, -1)))
            
            x = np.vstack(x)
            pd.DataFrame(x).to_csv(os.path.join(datpath, f'X{epoch:02d}.csv'), index=None, header=None)
            
            im = np.vstack(im)
            imout = np.zeros((100*256, 50*256, 3), dtype='uint8')
            for i in range(100):
                for j in range(50):
                    imout[256*i:256*(i+1), 256*j:256*(j+1), :] = im[i*50+j,:,:,::-1]
            cv2.imwrite(os.path.join(datpath, f'I{epoch:02d}.png'), imout)
        
        
        if np.mod(epoch+1, 1) == 0:
            manager.save()
            TestModel.save(f'Encoder{epoch:02d}', include_optimizer=False)
        
        if np.any(np.isnan(loss)):
            break
    
    return x, im

#%%
EPOCHS = 20
X, IM = fit(train_dataset, EPOCHS)


#%%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from umap import UMAP

U0 = UMAP(n_neighbors=15).fit(X)
U = U0.embedding_

plt.scatter(U[:,0],U[:,1],c='k')
ax = plt.gca()
for i in np.random.choice(IM.shape[0], 800, replace=False):
    im = cv2.resize(IM[i,:,:,:], (150,150))
    imagebox = OffsetImage(im, zoom=0.25)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, U[i,:2], xybox=(0, 0), xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)

