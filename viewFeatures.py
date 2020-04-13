
from sklearn.decomposition import PCA


#%%
def model():
    temp = Encoders[0]
    outputs = [temp.get_layer(index=i).output for i in [j for j,n in enumerate(temp.layers) if ('add' in n.name)]]
    return tf.keras.Model([temp.input], outputs)

M = model()


#%%
inp = []
test = []
for n in test_dataset.take(500):
    test.append([np.array(i) for i in M(n)])
    inp.append(np.array(n))
inp = np.vstack(inp)

out = [[] for _ in range(len(test[0]))]
for i in test:
    for n,j in enumerate(i):
        out[n].append(j)
test = [np.vstack(i) for i in out]
#%%
N = 9
F = 5
im = ((np.squeeze(np.concatenate(np.split(inp[:N,:,:,::-1], N), axis=2))+1)*127.5).astype('uint8')
IM = np.tile(im, (F,1,1))
cv2.imshow('im0',im)

#%%
x = test[3]
P = PCA(n_components=F).fit(np.reshape(x,(-1,x.shape[-1])))
print(P.explained_variance_ratio_)
c = P.components_

x1 = np.reshape(P.transform(np.reshape(x,(-1,x.shape[-1]))), list(x.shape[:3])+[F])

#%%
#i = np.arange(F)
i = np.argsort(x.std(axis=(0,1,2)))[-5:]
mask = np.concatenate(np.split(x[:N,:,:,i], N), axis=2)
mask = np.concatenate(np.split(mask, len(i), axis=3), axis=1)
mask = (mask-mask.min())/(mask.max()-mask.min())
mask = np.repeat(np.repeat(mask, 256//x.shape[1], 1), 256//x.shape[2], 2)[0,:,:,:]

cv2.imshow('im',(IM.astype('float')*mask).astype('uint8'))