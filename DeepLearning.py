############################################################################### Step 1 Import Modules
import numpy as np
import pandas as pd
import cv2 
import os 
import glob

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, relu, sigmoid
from losses import mse, mse_prime

#from keras.datasets import mnist
#from keras.utils import np_utils


############################################################################### Data Preparation
 
img_dir = "C:\\Hussain\\3rd Semester\\HRI\Assignment_3\\images" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img1 = cv2.imread(f1)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    data.append(img)


labels = pd.read_csv("labels.csv")
labels = np.array(labels)
labels.shape


data = np.array(data)
print(type(data))
print(data.shape)

data = data.flatten().reshape(20, 90000)
print(data.shape)

X = data
y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

###############################################################################

X_train = X_train.reshape(X_train.shape[0], 1, 300*300)
X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.reshape(X_test.shape[0], 1, 300*300)
X_test = X_test.astype('float32')
X_test /= 255

############################################################################### Step 3 Network
# Network
net = Network()
net.add(FCLayer(300*300, 100))               
net.add(ActivationLayer(tanh, relu))
net.add(FCLayer(100, 50))                   
net.add(ActivationLayer(tanh, relu))
net.add(FCLayer(50, 25))                   
net.add(ActivationLayer(tanh, relu))
net.add(FCLayer(25, 2))                    
net.add(ActivationLayer(tanh, sigmoid))

net.use(mse, mse_prime)
history = net.fit(X_train, y_train, epochs=20, learning_rate=0.01)

############################################################################### Step 4 Prediction
out = net.predict(X_test)
out = np.array(out)
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test)

###############################################################################
from sklearn.neural_network import MLPClassifier
X_train = X_train.reshape(300*300,X_train.shape[0])
y_train = y_train.reshape(1,y_train.shape[0])

mlp = MLPClassifier(hidden_layer_sizes=(5,5,5), activation='relu', solver='adam', max_iter=500, random_state=1,alpha=0.00095)
#X_train = X_train.T
#y_train = y_train.reshape(1,16)
mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
print(predict_test)