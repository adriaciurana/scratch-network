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
LR1 = LR1C()

net = Network()
net.load('example.h5')