import sys
import os
import numpy as np
import time
import h5py

sys.path.append(os.path.dirname(__file__)+"../../../")
from scratchnetwork import Network

# MNIST LOAD
from mnist import MNIST
mndata = MNIST('dataset/data')
#images_train, labels_train = mndata.load_training()
#images_train, labels_train = np.reshape(np.array(images_train), [-1, 28, 28]), np.array(labels_train)
images_test, labels_test = mndata.load_testing()
images_test, labels_test = np.reshape(np.array(images_test), [-1, 28, 28]), np.array(labels_test)

#images_train = (np.array(np.expand_dims(images_train, axis=-1), dtype=np.float64) - 128)/128
#labels_train = np.array([[float(m == b) for m in range(10)] for b in labels_train], dtype=np.float64)
images_test = ((np.array(np.expand_dims(images_test, axis=-1), dtype=np.float64) - 128)/128)
labels_test = np.array([[float(m == b) for m in range(10)] for b in labels_test], dtype=np.float64)
print(images_test.shape)

net = Network()
net.load('example.h5')

out = net.predict({'Input': images_test})
"""import matplotlib.pylab as plt
from random import shuffle
rrand = list(range(images_test.shape[0]))[:10]
shuffle(rrand)
for i in rrand:
	o = out['FC 2: Softmax'][i]
	o = np.argmax(o)
	# plot
	plt.title("number:"+str(o))
	plt.imshow(images_test[i][:,:,0])
	plt.colorbar()
	plt.show()"""

np.set_printoptions(threshold=np.nan)
a = np.expand_dims(np.argmax(out['FC 2: Softmax'], axis=1), axis=-1)
b = np.expand_dims(np.argmax(labels_test, axis=-1), axis=-1)
eq = a == b
accuracy_own = np.sum(eq)/eq.shape[0]

#print(np.hstack((a, b)))
from sklearn.metrics import confusion_matrix, accuracy_score
print('accuracy: ' + accuracy_score(b, a))
print('own accuracy: ' + accuracy_own)
print('Confusion Matrix:')
print(confusion_matrix(b, a))