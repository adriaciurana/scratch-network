import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network
from scratchnetwork.layers import Input
from scratchnetwork.layers import Conv2D
from scratchnetwork.layers import ReLU
from scratchnetwork.losses import MSE
from scratchnetwork.metrics import MRSE

# MNIST LOAD
from mnist import MNIST
mndata = MNIST('dataset/data')
images_train, labels_train = mndata.load_training()
images_train, labels_train = np.reshape(np.array(images_train), [-1, 28, 28]), np.array(labels_train)
images_test, labels_test = mndata.load_testing()
images_test, labels_test = np.reshape(np.array(images_test), [-1, 28, 28]), np.array(labels_test)

# Network
net = Network()
inputX = net.Node("Input", Input, [28, 28])
inputY = net.Node("Y", Input, [10])

B1 = net.Node("Block 1: Conv2D", Conv2D, 10)
B1relu = net.Node("Block 1: ReLU", ReLU)
B1max = net.Node("Block 1: Maxpooling", Pooling, "max")
B1drop = net.Node("Block 1: Dropout", Dropout, 0.25)




exit()

net = Network()
inputX = net.Node("Input", Input, [10])
inputY = net.Node("Y", Input, [1])

B = net.Node("B", FC, 10)
Br = net.Node("B_relu", ReLU)
C = net.Node("Output", FC, 1)

L1 = net.Node("MSE", MSE)
M1 = net.Node("MRSE", MRSE)

inputX.addNext(B)
B.addNext(Br)
Br.addNext(C)

L1.addPrev(C)
L1.addPrev(inputY)

M1.addPrev(C)
M1.addPrev(inputY)

net.compile(losses=[L1], metrics=[M1])
net.start(inputs=[inputX], outputs=[C])
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

# Llenamos
a = 2*(np.random.rand(1000, 10) - 0.5)
b = np.dot(a, np.random.rand(10, 1)) #np.dot(np.sign(a)*a**2, 100*np.random.rand(10, 1))

batch_index = 0
batch_size = 20
for i in range(100000):
	Xaux = a[batch_index:(batch_index + batch_size)]
	Yaux = b[batch_index:(batch_index + batch_size)]

	net.train_batch({'Input': Xaux}, {'Y': Yaux})

	batch_index += batch_size
	if batch_index >= a.shape[0]:
		batch_index = 0
	if i % 500 == 0:
		net.monitoring()
out = net.predict({'Input': a})
print(np.hstack((out['Output'], b)))