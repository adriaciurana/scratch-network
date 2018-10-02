import sys
import os
import numpy as np
import matplotlib.pylab as plt

sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network
from scratchnetwork.layers import Input
from scratchnetwork.layers import Conv2D
from scratchnetwork.layers import Pooling2D
from scratchnetwork.losses import MSE
from scratchnetwork.metrics import MRSE
from scratchnetwork.optimizers import SGD
from scipy import signal
from scipy import ndimage
from scipy import misc
net = Network()
inputX = net.Node("Input", Input, [20, 20, 1])
inputY = net.Node("Y", Input, [10, 10, 1])

N1 = net.Node("Block 1: Conv2D", Conv2D, 1, (3, 3), (1, 1), 'valid')
N2 = net.Node("Block 1: Pooling2D", Pooling2D, 'max', (5, 5), (1, 1), 'valid')
N3 = net.Node("Output", Conv2D, 1, (5, 5), (1, 1), 'valid')

L1 = net.Node("MSE", MSE)
M1 = net.Node("MRSE", MRSE)

inputX.addNext(N1)
N1.addNext(N2)
N2.addNext(N3)

L1.addPrev(N3)
L1.addPrev(inputY)

M1.addPrev(N3)
M1.addPrev(inputY)

net.compile(losses=[L1], metrics=[M1])
net.start(inputs=[inputX], outputs=[N3])
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

# Llenamos
a = 2*(np.random.rand(1000, 20, 20, 1) - 0.5)
b = 5*net.predict({'Input': a})['Output']
batch_size = 20
batch_index = 0

net.compile(losses=[L1], metrics=[M1], optimizer=SGD(lr=0.1, clip=1))
net.start(inputs=[inputX], outputs=[N3])

for i in range(10000):
	Xaux = a[batch_index:(batch_index + batch_size)]
	Yaux = b[batch_index:(batch_index + batch_size)]
	net.train_batch({'Input': Xaux}, {'Y': Yaux})

	batch_index += batch_size
	if batch_index >= a.shape[0]:
		batch_index = 0
	if i % 500 == 0:
		net.monitoring()
out = net.predict({'Input': a})
print(np.transpose(np.vstack((out['Output'].flatten(), b.flatten()))))
