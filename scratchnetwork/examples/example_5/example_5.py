import sys, os, time
import numpy as np
from mnist import MNIST
sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network
from scratchnetwork.layers import Input, FC, Conv2D, Pooling2D, DropOut, ReLU, Flatten, Softmax, OneHotDecode, Concat
from scratchnetwork.utils import Pipeline
from scratchnetwork.losses import SoftmaxCrossEntropy
from scratchnetwork.metrics import Accuracy
from scratchnetwork.optimizers import SGD, AdaGrad
from scratchnetwork.regularizators import L1 as LR1C
from scratchnetwork.callbacks import PrettyMonitor
LR1 = LR1C(0.0005)


# MNIST LOAD
from mnist import MNIST
mndata = MNIST('../datasets/mnist/data')
images_train, labels_train = mndata.load_training()
images_train, labels_train = np.reshape(np.array(images_train), [-1, 28, 28]), np.array(labels_train)

images_train = (np.array(np.expand_dims(images_train, axis=-1), dtype=np.float64) - 128)/128
labels_train = np.array(labels_train, dtype=np.int32).reshape(-1, 1)

# Network
net = Network()
inputX = net.Node(Input, "Input", [28, 28, 1])
inputY = net.Node(Input, "Label", [1])

o = net.Node(Conv2D, "Block 1: Conv2D", num_filters=32, kernel_size=(3,3), params={'regularizator': LR1})(inputX)
o = net.Node(ReLU, "Block 1: ReLU")(o)

def creator(net):
	i = net.Node(Conv2D, "Block 2: Conv2D", num_filters=64, kernel_size=(3,3), params={'regularizator': LR1})
	o = net.Node(ReLU, "Block 2: ReLU")(i)
	o = net.Node(Pooling2D, "Block 2: Maxpooling", "max", pool_size=(2, 2))(o)
	o = net.Node(DropOut, "Block 2: Dropout", 0.25)(o)
	o = net.Node(Flatten, "Block 2: Flatten")(o)
	return i, o
block = net.Node(Pipeline, "Pipeline", creator=creator)
block1 = block.copy(reuse=False)(o)
block2 = block.copy(reuse=False)(o)

o = net.Node(Concat, "Concat", axis=0)(block1, block2)
ço = net.Node(FC, "FC 1: FC", 128, params={'regularizator': LR1})(o)
o = net.Node(ReLU, "FC 1: ReLU")(o)
o = net.Node(DropOut, "FC 1: Dropout", 0.5)(o)

FC2 = net.Node(FC, "FC 2: FC ", 10, params={'regularizator': LR1})(o)
FC2softmax = net.Node(Softmax, "FC 2: Softmax")(FC2)
output = net.Node(OneHotDecode, "Output")(FC2softmax)

L1 = net.Node(SoftmaxCrossEntropy, "Cross Entropy")(FC2, inputY)
M1 = net.Node(Accuracy, "Accuracy")(output, inputY)

net.compile(inputs=[inputX], outputs=[output], losses=[L1], metrics=[M1], optimizer=SGD(lr=1e-2, clip_norm=None))
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

# Llenamos
batch_index = 0
batch_size = 128
epoch = 0

for i in range(300):
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

# Save
net.save("example.h5")

# Load
net2 = Network()
net2.load("example.h5")
