import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
#import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# import data
data = pd.read_csv("data.csv")
X = data[data.columns.difference(['diagnosis'])]
Y = data['diagnosis']
## normalize

for i in X.columns:
  X[i] =np.pi*(X[i]-X[i].min())/(X[i].max()-X[i].min())

number_of_qubits = len(X.columns)
number_of_layers = 5
dev = qml.device('default.qubit', wires=number_of_qubits )

@qml.qnode(dev)
def circuit( features, weights ):
    qml.AngleEmbedding(features, wires=range(number_of_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(number_of_qubits))
    return qml.expval(qml.PauliZ(0))

def variational_circuit(features, weights, bias):
    return circuit(features, weights)+bias

def loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p)**2 #binary cross entropy for genes
    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

def cost_function(weights, bias, X, Y):
    predictions= [variational_circuit(x,weights, bias) for x in X] 
    return loss(Y, predictions)


weights_init = 0.01 * np.random.randn(number_of_layers, number_of_qubits, requires_grad=True)
bias_init = np.array(0.01, requires_grad=True)
 
#optimization
opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

weights = weights_init
bias = bias_init


#BELOW HERE

num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
feats_train = X.iloc[list(index[:num_train]), range(number_of_qubits)] ##Change? 
Y_train = Y.iloc[list(index[:num_train])] ##??
feats_val = X.iloc[list(index[num_train:]), range(number_of_qubits)] ##Change?
Y_val = Y.iloc[list(index[num_train:])] ##?

# We need these later for plotting
X_train = X.iloc[list(index[:num_train]), :]
X_val = X.iloc[list(index[num_train:]), :]

print(data['diagnosis'].iloc(list(index[:num_train]))

for it in range(60):
    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    weights, bias, _, _ = opt.step(cost_function, weights, bias, feats_train_batch, Y_train_batch)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_circuit(f, weights, bias)) for f in feats_train]
    predictions_val = [np.sign(variational_circuit(f, weights, bias)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, cost(weights, bias, features, Y), acc_train, acc_val)
    )





