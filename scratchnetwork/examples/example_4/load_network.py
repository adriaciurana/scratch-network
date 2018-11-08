import sys
import os
import numpy as np
import time
import h5py

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
accuracy: 0.9572
Confusion Matrix:
[[ 972    0    1    0    0    1    4    1    1    0]
 [   0 1115    7    3    1    0    8    0    1    0]
 [  10    1  998   11    3    0    2    6    1    0]
 [   1    0    8  977    0   13    0    9    1    1]
 [   2    0    1    0  954    0    8    0    1   16]
 [   5    0    0    9    0  868    9    0    0    1]
 [  11    1    0    0    4    2  940    0    0    0]
 [   2    3   25    4    3    0    0  981    1    9]
 [  36    2   24   14    3   24   15   11  825   20]
 [   8    6    2   17   12   12    0   10    0  942]]
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
	plt.imshow(images_test[i][:,:,0])
	plt.colorbar()
	plt.show()

np.set_printoptions(threshold=np.nan)
a = out['Output']
b = labels_test
eq = a == b
accuracy = np.sum(eq)/eq.shape[0]
from sklearn.metrics import confusion_matrix
print('Accuracy: ' + str(accuracy))
print('Confusion Matrix:')
print(confusion_matrix(b, a))
