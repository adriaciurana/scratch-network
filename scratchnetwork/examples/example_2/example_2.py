import sys, os, time
import numpy as np
import matplotlib.pylab as plt
from scipy import signal, ndimage, misc
sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network
from scratchnetwork.layers import Input, Conv2D
from scratchnetwork.losses import MSE
from scratchnetwork.metrics import MRSE
from scratchnetwork.optimizers import SGD

net = Network()
inputX = net.Node("Input", Input, [10, 10, 1])
inputY = net.Node("Y", Input, [6, 6, 1])

B = net.Node("Output", Conv2D, 1, (5, 5), (1, 1), 'valid')

L1 = net.Node("MSE", MSE)
M1 = net.Node("MRSE", MRSE)

inputX.addNext(B)

L1.addPrev(B)
L1.addPrev(inputY)

M1.addPrev(B)
M1.addPrev(inputY)

optimizer=SGD(lr=1, mu=0, clip_norm=1)
net.compile(inputs=[inputX], outputs=[B], losses=[L1], metrics=[M1], optimizer=optimizer)
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

# Llenamos
a = 2*(np.random.rand(1000, 10, 10, 1) - 0.5)
b = np.zeros(shape=(1000, 6, 6, 1))
batch_size = 20

w1 = np.array([[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]])
w2 = np.transpose(w1)
s, k = 1, 2
w3 = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)]
w3 = np.outer(w3, w3)
for f in [w1, w2, w3]:
	a0 = time.time()
	batch_index = 0
	net.recompile()

	f *= 3
	ff = np.flipud(np.fliplr(f))
	for i in range(b.shape[0]):
		for d in range(a.shape[-1]):
			b[i,:,:,d] = 2 + signal.convolve2d(a[i,:,:,d], ff, 'valid')

	for i in range(100000):
		Xaux = a[batch_index:(batch_index + batch_size)]
		Yaux = b[batch_index:(batch_index + batch_size)]
		net.train_batch({'Input': Xaux}, {'Y': Yaux})

		batch_index += batch_size
		if batch_index >= a.shape[0]:
			batch_index = 0
		if i % 500 == 0:
			net.monitoring()
	print(time.time() - a0)
	out = net.predict({'Input': a})
	kernels = net.get_weights('Output').get('kernels')

	# plot
	plt.subplot(1,2,1)
	plt.imshow(kernels[0][0])
	plt.subplot(1,2,2)
	plt.imshow(f)
	plt.colorbar()
	plt.show()
	
	print(np.transpose(np.vstack((out['Output'].flatten(), b.flatten()))))
