import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__)+"../../")
from scratchnetwork import Network
from scratchnetwork.layers import Input
from scratchnetwork.layers import Conv2D
from scratchnetwork.losses import Loss
from scratchnetwork.metrics import Metric
from scipy import signal
from scipy import ndimage
from scipy import misc
net = Network()
inputX = net.Node("Input", Input, [10, 10, 1])
inputY = net.Node("Y", Input, [6, 8, 1])

B = net.Node("Output", Conv2D, 1, (5, 3))

L1 = net.Node("Loss", Loss)
M1 = net.Node("Metric", Metric)

inputX.addNext(B)

L1.addPrev(B)
L1.addPrev(inputY)

M1.addPrev(B)
M1.addPrev(inputY)

net.compile(losses=[L1], metrics=[M1])
net.start(inputs=[inputX], outputs=[B])
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

# Llenamos
a = 2*(np.random.rand(1000, 10, 10, 1) - 0.5)
b = np.zeros(shape=(1000, 6, 8, 1))

s, k = 1, 2
w = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)]
w = np.outer(w, w)
#w = w[:5, :3]
#w = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
w = np.array([[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2]])
w *= 3
print(w)
wf = np.flipud(np.fliplr(w))
for i in range(b.shape[0]):
	for d in range(a.shape[-1]):
		b[i,:,:,d] = signal.convolve2d(a[i,:,:,d], wf, 'valid')

batch_index = 0
batch_size = 20
for i in range(2000):
	Xaux = a[batch_index:(batch_index + batch_size)]
	Yaux = b[batch_index:(batch_index + batch_size)]
	net.train_batch({'Input': Xaux}, {'Y': Yaux})

	batch_index += batch_size
	if batch_index >= a.shape[0]:
		batch_index = 0
	if i % 500 == 0:
		net.monitoring()
out = net.predict({'Input': a})
kernels = net.get_weights('Output').get('kernels')
"""print(kernels.shape)
kernels = (1./(5*3))*np.sum(np.sum(kernels, axis=0), axis=0)
kernels = np.flipud(np.fliplr(kernels))
print(kernels)"""

print(kernels.shape)
import matplotlib.pylab as plt
plt.subplot(1,2,1)
plt.imshow(kernels[0][0])
plt.subplot(1,2,2)
plt.imshow(w)
plt.colorbar()
plt.show()
		

print(np.transpose(np.vstack((out['Output'].flatten(), b.flatten()))))