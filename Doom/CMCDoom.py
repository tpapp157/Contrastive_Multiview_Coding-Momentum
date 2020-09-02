import os

import tensorflow as tf
import numpy as np
import time
#import matplotlib.pyplot as plt
import cv2
#import boto3

from ConvNorm import ConvNorm
import pandas as pd

import albumentations

#%%
PATH = r'E:\Doom Images'
for _,_,files in os.walk(PATH):
    break
K0 = [os.path.join(PATH,f) for f in files]
#K0 = K0[:50001]

K1 = K0[1:]
K0 = K0[:-1]


if os.path.isfile('BayesSample.csv'):
    B = np.squeeze(np.array(pd.read_csv('BayesSample.csv', header=None, index_col=None)))
else:
    B = np.ones(len(K0), dtype='float')


# =============================================================================
# if os.path.isfile('sample.csv'):
#     sample = np.array(pd.read_csv('sample.csv', header=None, index_col=None))
#     K0 = [str(i[0]) for i in np.array(K0)[sample]]
#     K1 = [str(i[0]) for i in np.array(K1)[sample]]
# =============================================================================
#K = np.expand_dims(np.arange(len(K0)+1), 1)
print(len(K0))


#%%
BATCH_SIZE = 20
PATCH_SIZE = 256

def ImgAug(i0):
    Aug = albumentations.Compose([albumentations.HorizontalFlip(p=0.25),
                                  #albumentations.RandomBrightness(p=0.25),
                                  albumentations.RandomGamma((80, 120), p=0.25),
                                  albumentations.HueSaturationValue(hue_shift_limit=20, p=0.25),
                                  #albumentations.CLAHE(p=0.25),
                                  albumentations.Blur(p=0.25),
                                  albumentations.ShiftScaleRotate(shift_limit=0, rotate_limit=20, p=0.25),
                                  albumentations.augmentations.transforms.ToGray(p=0.25)
                                  ], p=0.50)
    
    if type(i0)!=np.ndarray:
        i0 = i0.numpy()
    i0 = Aug(image=i0.astype('uint8'))['image']
    i0 = i0.astype('float32') / 127.5 - 1
    
    return i0
    

def cvLoad(f1):
    IM0 = cv2.imread(K0[f1.numpy()])
    IM1 = cv2.imread(K1[f1.numpy()])
    H, W, _ = IM0.shape
    
    n = int(2**np.random.uniform(9, 10))
    h = np.random.choice(H-n)
    w = np.random.choice(W-n)
    
    im0lg = IM0[h:h+n, w:w+n, ::-1]
    im1lg = IM1[h:h+n, w:w+n, ::-1]
    
    n = int(n//2)
    h,w = np.random.choice(im0lg.shape[0]-n, 2)
    im0md = im0lg[h:h+n, w:w+n, :]
    
    n = int(n//2)
    h,w = np.random.choice(im0md.shape[0]-n, 2)
    im0sm = im0md[h:h+n, w:w+n, :]
    
    im0lg = ImgAug(cv2.resize(im0lg, (256,256), interpolation=cv2.INTER_AREA))
    im0md = ImgAug(cv2.resize(im0md, (256,256), interpolation=cv2.INTER_AREA))
    im0sm = ImgAug(cv2.resize(im0sm, (256,256), interpolation=cv2.INTER_AREA))
    im1lg = ImgAug(cv2.resize(im1lg, (256,256), interpolation=cv2.INTER_AREA))
    
    return im0lg, im0md, im0sm, im1lg


def pf_load(f1):
    im0, im1, im2, im3 = tf.py_function(cvLoad, [f1], [tf.float32, tf.float32, tf.float32,  tf.float32])
    im0.set_shape([None, None, 3])
    im1.set_shape([None, None, 3])
    im2.set_shape([None, None, 3])
    im3.set_shape([None, None, 3])
    return im0, im1, im2, im3, f1




def npSample():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    
    #k = np.arange(len(K0)+1)
    #np.random.shuffle(k)
    
    b = (B-B.min())/(B.max()-B.min())
    b = b + np.median(b)
    
    
    k = np.random.choice(len(K0), len(K0), replace=True, p=b/np.sum(b))
    
    for i in k:
        
        D = np.abs(i - np.arange(len(K0)))
        n = 50
        D = 1 - sigmoid((D-n) / (n/5))
        D[i] = 0
        
        D = D * b
        
        D = D / D.sum()
        
        temp = np.random.choice(len(K0), BATCH_SIZE-1, replace=False, p=D)
        temp = np.hstack([i, temp])
        for j in temp:
            yield j



#%%
#train_dataset = tf.data.Dataset.from_tensor_slices((K0, K1)).shuffle(len(K0))
#train_dataset = tf.data.Dataset.from_tensor_slices((K0, K1)).shuffle(500)


train_dataset = tf.data.Dataset.from_generator(npSample, tf.int32)
train_dataset = train_dataset.map(pf_load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)


#%%
def test_load(f):
    IM0 = cv2.imread(f.numpy().decode("utf-8"))
    H, W, _ = IM0.shape
    
    n = 512
    h = int((H-n)/2)
    w = int((W-n)/2)
    im = IM0[h:h+n, w:w+n, ::-1]
    im = cv2.resize(im, (256,256), interpolation=cv2.INTER_AREA).astype('float32') / 127.5 - 1
    return im

def pf_test_load(f):
    im0 = tf.py_function(test_load, [f], tf.float32)
    im0.set_shape([None, None, 3])
    return im0


test_dataset = tf.data.Dataset.from_tensor_slices(K0).shuffle(len(K0)).take(10000)
test_dataset = test_dataset.map(pf_test_load, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)


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
    

#%%
OUTPUT_CHANNELS = 128

def FeatureExtractor(in_channels, channels=32, blocks=5):
    x_init = tf.keras.layers.Input(shape=[None, None, in_channels])
    x = conv(x_init, channels, kernel=5, stride=2)
    
    for i in range(blocks):
        x = resblock(x, stride=2)
        x = resblock(x)
    
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
    x = tf.nn.leaky_relu(x, 0.2)
    x = ConvNorm(in_channels, kernel_size=1)(x)
    
    c = int(h*n)
    q = tf.keras.layers.Dense(units=c)(tf.math.reduce_mean(x, axis=(1,2)))
    q = tf.split(q, h, axis=1)
    
    h0 = tf.shape(x)[1]
    w0 = tf.shape(x)[2]
    loc = tf.tile(tf.expand_dims(tf.concat([tf.expand_dims(tf.repeat(tf.expand_dims(tf.linspace(0.0, 1.0, h0), 1), w0, axis=1), 2), tf.expand_dims(tf.repeat(tf.expand_dims(tf.linspace(0.0, 1.0, w0), 0), h0, axis=0), 2)], 2), 0), (b, 1, 1, 1))
    x = tf.concat([x, loc], axis=3)
    
    k = tf.split(conv(x, c, 1), h, axis=3)
    k = [tf.reshape(i, (b, -1, n)) for i in k]
    v = tf.split(conv(x, c, 1), h, axis=3)
    v = [tf.reshape(i, (b, -1, n)) for i in v]
    
    
    k = [tf.nn.softmax(tf.matmul(q[i], tf.transpose(k[i], (0,2,1))) / tf.math.sqrt(tf.cast(n, tf.float32))) for i in range(len(k))]
    x = [tf.matmul(k[i], v[i]) for i in range(len(k))]
    x = tf.reshape(tf.reduce_mean(tf.concat(x, axis=2), axis=1), (-1, h*n))
    #x = tf.reduce_mean(tf.concat(x, axis=1), axis=1)
    x = tf.keras.layers.Dense(units=out_channels)(x)
    
    return tf.keras.Model(inputs=x_init, outputs=x)


Encoder = FeatureExtractor(3, channels=48, blocks=5)
i = Encoder.get_layer(index=-1).output_shape[0][-1]

#Classifiers = [Classifier(in_channels=i, out_channels=OUTPUT_CHANNELS) for _ in range(len(tf.data.experimental.get_structure(train_dataset)))]
Classifiers = [Att_Classifier(in_channels=i, out_channels=OUTPUT_CHANNELS, n=64, h=8) for _ in range(len(tf.data.experimental.get_structure(train_dataset))-1)]

combinations = []
combinations = [(0,1),(0,3),
                (1,0),(1,2),
                (2,1),
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
        
        
    def call(self, X, Mem):
        batch = X[0].shape[0]
        
        out0 = []
        Xn = [tf.linalg.norm(i, axis=1, keepdims=True) for i in X]
        Mn = [tf.repeat(tf.linalg.norm(i, axis=2), batch, axis=0) for i in Mem]
        for n,a in enumerate(self.combinations):
            i,j = a
            
            M = tf.concat((tf.expand_dims(X[j], 1), tf.repeat(Mem[j], batch, axis=0)), axis=1, name=f'concat_{i}{j}')
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
        
        return out0, out1


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
Encoder_ema = tf.keras.models.clone_model(Encoder)
Encoder_ema.trainable = False
#Classifiers_ema = [tf.keras.models.clone_model(i) for i in Classifiers]
Classifiers_ema = [Att_Classifier(in_channels=i, out_channels=OUTPUT_CHANNELS, n=64, h=8) for _ in range(len(tf.data.experimental.get_structure(train_dataset))-1)]
for i in range(len(Classifiers_ema)):
    for j,_ in enumerate(Classifiers_ema[i].variables):
        Classifiers_ema[i].variables[j].assign(Classifiers[i].variables[j].value())
        
optimizer = tf.keras.optimizers.Adam(1e-4, 0.99, clipnorm=100)


checkpoint_dir = r'ckpt'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(Encoder=Encoder, Encoder_ema=Encoder_ema, optimizer=optimizer)
checkpoint.Classifiers=Classifiers
checkpoint.Classifiers_ema=Classifiers_ema

manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5, keep_checkpoint_every_n_hours=2)
status = checkpoint.restore(manager.latest_checkpoint)
#status.assert_consumed()


for i in range(len(Classifiers_ema)):
    Classifiers_ema[i].trainable=False


#%%
@tf.function
def train_step(inputs, Mem):
    batch = inputs[0].shape[0]
    with tf.GradientTape() as grad_tape:
        
        x0 = []
        y = []
        for i in range(len(inputs)):
            x0.append(Encoder(inputs[i]))
            y.append(tf.stop_gradient(Encoder_ema(inputs[i])))
        
        x = [Classifiers[i](x0[i]) for i in range(len(x0))]
        y = [Classifiers_ema[i](y[i]) for i in range(len(y))]
        
        Mem = [tf.expand_dims(i, axis=0) for i in Mem]
        l_inter, l_intra = Memory(x, Mem)
        loss = [NCEloss(i, Q) for i in l_inter] + [NCEloss(i, batch) for i in l_intra]
        if tf.math.reduce_any([(tf.math.is_nan(i) or tf.math.is_inf(i)) for i in loss]):
            tf.print(loss)
        Loss = tf.math.reduce_sum(loss)
        
    V = Encoder.trainable_variables
    for i in set(Classifiers):
        V += i.trainable_variables
    grad = grad_tape.gradient(Loss, V)
    #if tf.math.is_inf(tf.linalg.global_norm(grad)):
    grad = [tf.clip_by_value(i, -1e10, 1e10) for i in grad]
    optimizer.apply_gradients(zip(grad, V))
    
    m = 0.99
    for j in range(len(Encoder.variables)):
        Encoder_ema.variables[j] = m*Encoder_ema.variables[j] + (1-m)*Encoder.variables[j]
    for i in range(len(Classifiers)):
        for j in range(len(Classifiers[i].variables)):
            Classifiers_ema[i].variables[j] = m*Classifiers_ema[i].variables[j] + (1-m)*Classifiers[i].variables[j]
    
    return y, loss, x0[0]


def fit(train_ds, epochs):
    datpath = r'data'
    if not os.path.isdir(datpath):
        os.makedirs(datpath)
    
    Mem = [[] for _ in range(len(Classifiers))]
    for inputs in train_ds.take(np.ceil(Q/BATCH_SIZE)):
        for i in range(len(Classifiers)):
            Mem[i] += [Classifiers_ema[i](Encoder_ema(inputs[i])).numpy()]
    Mem = [np.vstack(i)[-Q:,:] for i in Mem]
    
    
    for epoch in range(epochs):
        start = time.time()
        
        # Train
        loss = []
        x = []
        im = []
        c = 0
        for inputs in train_ds:
            c+=1
            nMem, l, emb = train_step(inputs[:-1], Mem)
            
            for i in range(len(Mem)):
                Mem[i] = np.vstack([Mem[i], nMem[i].numpy()])[-Q:,:]
            
            loss.append(np.array(l))
            im.append(np.array((inputs[0]+1)*127.5).astype('uint8'))
            x.append(np.mean(np.array(emb), axis=(1,2)))
            
            i = inputs[-1].numpy().astype('int')
            #print(i)
            B[i] = B[i]*0.8 + np.mean(np.array(l))*0.2
            
            if c*BATCH_SIZE==20000:
                break
        
        print('Epoch {} took {} min'.format(epoch, np.round((time.time()-start)/60, 2)))
        print(np.median(loss, axis=0))
        
        pd.DataFrame(B).to_csv('BayesSample.csv', index=None, header=None)
        
        loss = np.array(loss)
        pd.DataFrame(loss).to_csv(os.path.join(datpath, f'L{epoch:02d}.csv'), index=None, header=None)
        
        x = np.vstack(x[-int(np.ceil(10000/BATCH_SIZE)):])[-10000:,:]
        im = np.vstack(im[-int(np.ceil(2000/BATCH_SIZE)):])[-2000:,:,:,:]
        pd.DataFrame(x).to_csv(os.path.join(datpath, f'Xt{epoch:02d}.csv'), index=None, header=None)
        
        imout = np.zeros((40*256, 50*256, 3), dtype='uint8')
        for i in range(40):
            for j in range(50):
                imout[256*i:256*(i+1), 256*j:256*(j+1), :] = im[i*50+j,:,:,::-1]
        cv2.imwrite(os.path.join(datpath, f'It{epoch:02d}.png'), imout)
        
        
        if (epoch+1) % 5 == 0 or epoch+1==epochs:
            im = []
            x = []
            for inputs in test_dataset:
                emb = Encoder(inputs)
                
                im.append(np.array((inputs+1)*127.5).astype('uint8'))
                x.append(np.mean(np.array(emb), axis=(1,2)))
                #x.append(np.reshape(np.array(emb), (BATCH_SIZE, -1)))
            
            x = np.vstack(x)
            im = np.vstack(im[-int(np.ceil(2000/BATCH_SIZE)):])[-2000:,:,:,:]
            pd.DataFrame(x).to_csv(os.path.join(datpath, f'X{epoch:02d}.csv'), index=None, header=None)
            
            imout = np.zeros((40*256, 50*256, 3), dtype='uint8')
            for i in range(40):
                for j in range(50):
                    imout[256*i:256*(i+1), 256*j:256*(j+1), :] = im[i*50+j,:,:,::-1]
            cv2.imwrite(os.path.join(datpath, f'I{epoch:02d}.png'), imout)
            
        
        if (epoch+1) % 5 == 0 or epoch+1==epochs:
            manager.save()
            if not os.path.isdir('encoders'):
                os.makedirs('encoders')
            Encoder.save(f'encoders\Encoder{epoch:02d}', include_optimizer=False)
        
        if np.any(np.isnan(loss)):
            break
    
    return x, im

#%%
EPOCHS = 60
X, IM = fit(train_dataset, EPOCHS)
#Encoder.save(f'Encoder{EPOCHS:02d}', include_optimizer=False)


#%%
import matplotlib.pyplot as plt
from umap import UMAP

U0 = UMAP(n_neighbors=20).fit(X)
U = U0.embedding_
plt.scatter(U[:,0],U[:,1],c='k')


#%%
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

ax = plt.gca()
for i in np.random.choice(IM.shape[0], 800, replace=False):
    im = cv2.resize(IM[i,:,:,:], (150,150))
    imagebox = OffsetImage(im, zoom=0.25)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, U[i,:2], xybox=(0, 0), xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)

