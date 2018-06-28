import sys
import os
import numpy as np
import time
import h5py

sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network
from scratchnetwork.layers import Input
from scratchnetwork.layers import FC
from scratchnetwork.layers import Conv2D
from scratchnetwork.layers import Pooling2D
from scratchnetwork.layers import DropOut
from scratchnetwork.layers import ReLU
from scratchnetwork.layers import Flatten
from scratchnetwork.layers import Softmax
from scratchnetwork.losses import SoftmaxCrossEntropy
from scratchnetwork.metrics import Accuracy
from scratchnetwork.optimizers import SGD
from scratchnetwork.regularizators import L1 as LR1C
from scratchnetwork.layers import OneHotDecode
LR1 = LR1C(0.0005)


# MNIST LOAD
from mnist import MNIST
mndata = MNIST('dataset/data')
images_train, labels_train = mndata.load_training()
images_train, labels_train = np.reshape(np.array(images_train), [-1, 28, 28]), np.array(labels_train)

images_train = (np.array(np.expand_dims(images_train, axis=-1), dtype=np.float64) - 128)/128
labels_train = np.array(labels_train, dtype=np.int32).reshape(-1, 1)
#labels_train = np.array([[float(m == b) for m in range(10)] for b in labels_train], dtype=np.float64)

# Network
net = Network()
inputX = net.Node("Input", Input, [28, 28, 1])
inputY = net.Node("Label", Input, [1])

o = net.Node("Block 1: Conv2D", Conv2D, num_filters=32, kernel_size=(3,3), params={'regularizator': LR1})(inputX)
o = net.Node("Block 1: ReLU", ReLU)(o)

o = net.Node("Block 2: Conv2D", Conv2D, num_filters=64, kernel_size=(3,3), params={'regularizator': LR1})(o)
o = net.Node("Block 2: ReLU", ReLU)(o)
o = net.Node("Block 2: Maxpooling", Pooling2D, "max", pool_size=(2, 2))(o)
o = net.Node("Block 2: Dropout", DropOut, 0.25)(o)
o = net.Node("Block 2: Flatten", Flatten)(o)

o = net.Node("FC 1: FC", FC, 128, params={'regularizator': LR1})(o)
o = net.Node("FC 1: ReLU", ReLU)(o)
o = net.Node("FC 1: Dropout", DropOut, 0.5)(o)

FC2 = net.Node("FC 2: FC ", FC, 10, params={'regularizator': LR1})(o)
FC2softmax = net.Node("FC 2: Softmax", Softmax)(FC2)
output = net.Node("Output", OneHotDecode)(FC2softmax)

L1 = net.Node("Cross Entropy", SoftmaxCrossEntropy)(FC2, inputY)
M1 = net.Node("Accuracy", Accuracy)(output, inputY)

net.compile(losses=[L1], metrics=[M1], optimizer=SGD(lr=1e-2, clip=None))
net.start(inputs=[inputX], outputs=[output])
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

# Llenamos
batch_index = 0
batch_size = 128
epoch = 0

for i in range(3):
	Xaux = images_train[batch_index:(batch_index + batch_size)]
	Yaux = labels_train[batch_index:(batch_index + batch_size)]

	t = time.time()
	net.train_batch({'Input': Xaux}, {'Label': Yaux})
	batch_index += batch_size
	if batch_index >= images_train.shape[0]:
		batch_index = 0
		epoch += 1
	
	net.monitoring()
	print(str(batch_index) + "/" + str(epoch))
	print('-----'+ str(time.time() - t) +'------')


#print(net.save("example.h5"))