import sys, os, time
import numpy as np
from mnist import MNIST
sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network, Initializer, L1 as LR1C
from scratchnetwork.layers import Input, FC, Conv2D, Pooling2D, DropOut, ReLU, Flatten, Softmax, OneHotDecode
from scratchnetwork.losses import SoftmaxCrossEntropy
from scratchnetwork.metrics import Accuracy
from scratchnetwork.optimizers import SGD, AdaGrad
from scratchnetwork.callbacks import PrettyMonitor


LR1 = LR1C(0)

# MNIST LOAD
mndata = MNIST('../datasets/mnist/data')
images_train, labels_train = mndata.load_training()
images_train, labels_train = np.reshape(np.array(images_train), [-1, 28, 28]), np.array(labels_train)

images_train = (np.array(np.expand_dims(images_train, axis=-1), dtype=np.float64) - 128)/128
labels_train = np.array(labels_train, dtype=np.int32).reshape(-1, 1)

# Network
net = Network()
inputX = net(Input, "Input", [28, 28, 1])
inputY = net(Input, "Label", [1])

o = net(Conv2D, "Block 1: Conv2D", num_filters=32, kernel_size=(3,3), initializer=Initializer('he'), regularizator=LR1)(inputX)
o = net(ReLU, "Block 1: ReLU")(o)

o = net(Conv2D, "Block 2: Conv2D", num_filters=64, kernel_size=(3,3), initializer=Initializer('he'), regularizator=LR1)(o)
o = net(ReLU, "Block 2: ReLU")(o)
o = net(Pooling2D, "Block 2: Maxpooling", "max", pool_size=(2, 2))(o)
o = net(DropOut, "Block 2: Dropout", 0.25)(o)
o = net(Flatten, "Block 2: Flatten")(o)

o = net(FC, "FC 1: FC", 128, params={'regularizator': LR1})(o)
o = net(ReLU, "FC 1: ReLU")(o)
o = net(DropOut, "FC 1: Dropout", 0.5)(o)

FC2 = net(FC, "FC 2: FC", 10, params={'regularizator': LR1})(o)
FC2softmax = net(Softmax, "FC 2: Softmax")(FC2)
output = net(OneHotDecode, "Output")(FC2softmax)

L1 = net(SoftmaxCrossEntropy, "Cross Entropy")(FC2, inputY)
M1 = net(Accuracy, "Accuracy")(output, inputY)

net.compile(inputs=[inputX], outputs=[output], losses=[L1], metrics=[M1], optimizer=AdaGrad(lr=1e-2, clip_norm=None))
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")

"""# Llenamos
batch_index = 0
batch_size = 128
epoch = 0

for i in range(10000):
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
	
"""
iterations = {
	'training': 1000,
	'validation': 1,
}
callbacks = [PrettyMonitor(PrettyMonitor.TRAINING, 1), PrettyMonitor(PrettyMonitor.VALIDATION)]
net.fit(
	X={'Input': images_train}, 
	Y={'Label': labels_train}, 
	epochs=10, 
	batch_size=128, 
	Xval={'Input': images_train}, 
	Yval={'Label': labels_train}, 
	shuffle=True, 
	iterations=iterations,
	callbacks=callbacks)
net.save("example.h5", freeze=True)
