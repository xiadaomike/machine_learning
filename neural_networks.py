import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import scipy.io as sio
import scipy.stats as sstats
import signal
import sys
from sklearn import preprocessing
from sklearn.utils import shuffle

# Configuration
sshot_name = 'snapshot.npz'
if os.path.isfile(sshot_name):
    snapshot = np.load(sshot_name)
    W1, W2 = snapshot['W1'], snapshot['W2']
    P1, P2 = snapshot['P1'], snapshot['P2']
    iteration = snapshot['iteration']
    eta = snapshot['eta']
    train_X, train_Y = snapshot['train_X'], snapshot['train_Y']
    #val_X, val_Y = snapshot['val_X'], snapshot['val_Y']
    test_X = snapshot['test_X']
else:
    W1 = 0.01*np.random.randn(785, 200)
    W2 = 0.01*np.random.randn(201, 10)
    P1 = np.zeros([785, 200])
    P2 = np.zeros([201, 10])
    iteration = [0]
    eta = [0.001]

    # Data preprocessing
    train_data = sio.loadmat('./dataset/train.mat')
    test_data = sio.loadmat('./dataset/test.mat')
    
    train_X = train_data['train_images']
    train_X = train_X.swapaxes(1, 2).swapaxes(0, 1).reshape(60000,
            28*28).astype(float)
    preprocessing.normalize(train_X, copy=False)
    
    train_Y = train_data['train_labels'].reshape(60000)
    train_X, train_Y = shuffle(train_X, train_Y)
    
    scaler = preprocessing.StandardScaler(copy=False).fit(train_X)
    scaler.transform(train_X)
    
    #val_set = sio.loadmat('val.mat')
    #val_X_tmp = val_set['test_images'].astype(float)
    #val_Y = val_set['test_labels'].reshape(10000)
    
    #val_X = np.empty([10000, 784])
    #for i, img in enumerate(val_X_tmp):
    #    val_X[i] = np.fliplr(np.rot90(img.reshape(28, 28), 3)).reshape(1, 28*28)
    #
    #val_X = preprocessing.normalize(val_X)
    #scaler.transform(val_X)
    
    #val_X, val_Y = shuffle(val_X, val_Y)

    test_X = test_data['test_images']
    test_X = test_X.swapaxes(1, 2).swapaxes(0, 1).reshape(10000,
            28*28).astype(float)
    preprocessing.normalize(test_X, copy=False)
    scaler.transform(test_X)

def signal_handler(signal, frame):
    np.savez('snapshot.npz', W1=W1, W2=W2, P1=P1, P2=P2, iteration=iteration,
            eta=eta, train_X=train_X, train_Y=train_Y, test_X=test_X)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def d_tanh(z):
    np.tanh(z, out=z)
    np.square(z, out=z)
    np.subtract(1.0, z, out=z)
    return z

def sig(z, o):
    np.multiply(z, -1.0, out=o)
    np.exp(o, out=o)
    np.add(o, 1.0, out=o)
    np.divide(1.0, o, out=o)

dsig_tmp = np.empty(10)
def d_sig(z): 
    sig(z, z)
    np.subtract(1, z, out=dsig_tmp)
    np.multiply(z, dsig_tmp, out=z)
    return z

def train_neural_network(X, Y, loss_grad):
    accs, itr_r = [], []
    X = np.insert(X, 0, 1, axis=1)
    Z_hid = np.empty(200)
    Y_hid = np.empty(200)
    Y_hid_wb = np.empty(201)
    Z_out = np.empty(10)
    Y_out = np.empty(10)

    loss_g = np.empty(10)
    diag_dsig = np.empty([10, 10])
    dz_out = np.empty(10)
    dw2 = np.empty([201, 10])

    diag_dtanh = np.empty([200, 200])
    dot_tmp = np.empty([200, 10])
    dz_hid = np.empty(200)
    dw1 = np.empty([785, 200])

    last = iteration[0] + 200000
    while iteration[0] < last:
        sample_i = np.random.randint(0, X.shape[0])
        x, y = X[sample_i], Y[sample_i]
        zeros = np.zeros(10)
        zeros[y] = 1
        y = zeros

        np.dot(x, W1, out=Z_hid)                # 1*785, 785*200, 1*200
        np.tanh(Z_hid, out=Y_hid)               # 1*200

        Y_hid_wb[0] = 1.0
        Y_hid_wb[1:] = Y_hid                    # 1*201
        np.dot(Y_hid_wb, W2, out=Z_out)         # 1*201, 201*10, 1*10
        sig(Z_out, Y_out)

        # 10*10, 10*1, 10*1
        diag_dsig = np.diag(d_sig(Z_out))
        np.dot(diag_dsig, loss_grad(Y_out, y, loss_g), out=dz_out)
        # 1*201, 1*10, 201*10
        np.outer(Y_hid_wb, dz_out.T, out=dw2)

        # 200*200, 200*10, 10*1, 200*1
        diag_dtanh = np.diag(d_tanh(Z_hid))
        np.dot(diag_dtanh, W2[1:], out=dot_tmp)
        np.dot(dot_tmp, dz_out, out=dz_hid)
        # 1*785, 1*200, 785*200
        np.outer(x, dz_hid.T, out=dw1)

        np.multiply(0.9, P1, out=P1)
        np.subtract(P1, np.multiply(eta[0], dw1, out=dw1), out=P1)
        np.multiply(0.9, P2, out=P2)
        np.subtract(P2, np.multiply(eta[0], dw2, out=dw2), out=P2)
        np.add(W1, P1, out=W1)
        np.add(W2, P2, out=W2)

        iteration[0] += 1

        if iteration[0] % 1000 == 0:
            #preds = predict_neural_network(W1, W2, val_X)
            #acc = (preds == val_Y).sum()/float(len(val_Y))
            #print iteration[0], acc, eta[0]
            preds = predict_neural_network(W1, W2, train_X)
            acc = (preds == train_Y).sum()/float(len(train_Y))
            accs.append(acc)
            itr_r.append(iteration[0])
            print iteration[0], acc, eta[0]

        if iteration[0] % (5*50000) == 0:
            eta[0] *= 0.8

    np.savez('snapshot.npz', W1=W1, W2=W2, P1=P1, P2=P2, iteration=iteration,
            eta=eta, train_X=train_X, train_Y=train_Y, test_X=test_X)
    return W1, W2, accs, itr_r

def predict_neural_network(W1, W2, test_X):
    test_X = np.insert(test_X, 0, 1, axis=1)

    # n*785, 785*200, n*200
    Z_hid = test_X.dot(W1)
    Y_hid = np.tanh(Z_hid)

    Y_hid = np.insert(Y_hid, 0, 1, axis=1)
    # n*201, 201*10, n*10
    Z_out = Y_hid.dot(W2)
    sig(Z_out, Z_out)

    return np.argmax(Z_out, axis=1)

def generate_loss_func(variety):
    def loss_grad(preds, truth, o):
        if variety == "msq":
            np.subtract(preds, truth, out=o)
            return o
        elif variety == "c_ent":
            return -truth/preds + (1-truth)/(1-preds)
    return loss_grad

# Training & Testing
loss_grad = generate_loss_func("c_ent")
W1, W2, accs, itr_r = train_neural_network(train_X, train_Y, loss_grad)
plt.plot(itr_r, accs)
plt.show()

preds = predict_neural_network(W1, W2, test_X)

# Output
with open('digit_preds.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    index = 1
    for i in range(len(preds)):
        writer.writerow({'Id': str(i+1), 'Category': str(preds[i])})
        index += 1
