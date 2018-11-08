import sys, os, time, h5py
import numpy as np

sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network

# MNIST LOAD
from mnist import MNIST
mndata = MNIST('../datasets/mnist/data')
images_test, labels_test = mndata.load_testing()
images_test, labels_test = np.reshape(np.array(images_test), [-1, 28, 28]), np.array(labels_test)

images_test = (np.array(np.expand_dims(images_test, axis=-1), dtype=np.float64) - 128)/128
labels_test = np.array(labels_test, dtype=np.int32).reshape(-1, 1)

net = Network()
# DESCARGAR LOS PESOS EN: https://drive.google.com/file/d/1DeewIZLeeEUqVIsWrcPwc1Y3zr7eFUDA/view?usp=sharing
"""
Accuracy: 0.9903
Confusion Matrix:
[[ 976    0    0    0    0    0    2    1    1    0]
 [   0 1130    2    1    0    1    1    0    0    0]
 [   2    1 1023    0    1    0    0    5    0    0]
 [   1    0    1 1001    0    2    0    3    2    0]
 [   0    0    0    0  973    0    1    0    2    6]
 [   2    0    0    3    0  884    2    0    0    1]
 [   6    3    0    0    1    1  947    0    0    0]
 [   0    1    3    2    1    0    0 1018    2    1]
 [   3    1    2    2    0    0    0    2  962    2]
 [   2    1    0    3    3    4    0    6    1  989]]
"""
net.load('example.h5')
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")
out = net.predict({'Input': images_test})
import matplotlib.pylab as plt
from random import shuffle
rrand = list(range(images_test.shape[0]))[:10]
shuffle(rrand)
for i in rrand:
	o = out['Output'][i, 0]

	# plot
	plt.title("number: "+str(o))
	plt.imshow(images_test[i][:,:,0], cmap='gray')
	plt.show(block = False)
	while plt.waitforbuttonpress() is None:
		pass

np.set_printoptions(threshold=np.nan)
a = out['Output']
b = labels_test
eq = a == b
accuracy = np.sum(eq)/eq.shape[0]
from sklearn.metrics import confusion_matrix
print('Accuracy: ' + str(accuracy))
print('Confusion Matrix:')
print(confusion_matrix(b, a))
