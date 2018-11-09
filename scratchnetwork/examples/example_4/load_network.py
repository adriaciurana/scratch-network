import sys, os, time, h5py
import numpy as np
from mnist import MNIST
import matplotlib.pylab as plt
from random import shuffle
sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network

num_examples = 1000

# MNIST LOAD
mndata = MNIST('../datasets/mnist/data')
images_test, labels_test = mndata.load_testing()
images_test, labels_test = np.reshape(np.array(images_test), [-1, 28, 28]), np.array(labels_test)

images_test = (np.array(np.expand_dims(images_test, axis=-1), dtype=np.float64) - 128)/128
labels_test = np.array(labels_test, dtype=np.int32).reshape(-1, 1)

net = Network()
# DESCARGAR LOS PESOS EN: https://drive.google.com/file/d/1DeewIZLeeEUqVIsWrcPwc1Y3zr7eFUDA/view?usp=sharing
"""
Accuracy: 0.9909
Confusion Matrix:
[[ 976    0    1    0    0    0    1    1    1    0]
 [   0 1130    2    1    0    0    2    0    0    0]
 [   1    0 1027    0    0    0    0    4    0    0]
 [   0    0    3 1004    0    1    0    0    2    0]
 [   0    0    1    0  972    0    1    0    1    7]
 [   2    0    0    3    0  883    2    0    2    0]
 [   5    2    0    0    1    3  944    0    3    0]
 [   1    0    9    2    0    0    0 1013    1    2]
 [   2    0    1    1    1    0    1    1  965    2]
 [   1    1    0    0    2    3    0    4    3  995]]
"""
net.load('example.h5')
print('Loaded...')
net.plot(os.path.basename(sys.argv[0]).split(".")[0]+".png")
out = net.predict({'Input': images_test[:num_examples]})
rrand = list(range(images_test.shape[0]))[:10]
shuffle(rrand)
for i in rrand:
	o = out['Output'][i, 0]

	# plot
	plt.title("number: "+str(o))
	plt.imshow(images_test[i][:,:,0], cmap='gray')
	plt.draw()
	plt.show(block = False)
	while plt.waitforbuttonpress(0) is None:
		pass

"""np.set_printoptions(threshold=np.nan)
a = out['Output']
b = labels_test
eq = a == b
accuracy = np.sum(eq)/eq.shape[0]
from sklearn.metrics import confusion_matrix
print('Accuracy: ' + str(accuracy))
print('Confusion Matrix:')
print(confusion_matrix(b, a))"""