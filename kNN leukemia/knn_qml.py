
import sys
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df  = pd.read_csv("Leukemia_GSE9476.csv")
types = df["type"]

labels_dict = {"AML" : 1, "Bone_Marrow_CD34" : 0, "Bone_Marrow" : 2, "PB" : 3, "PBSC_CD34" : 4}

labels = np.zeros(len(types), dtype=int)
labels_m = np.zeros(len(types), dtype=int)

for i in range(len(labels)):
    for type in labels_dict.keys():
        if types[i] == type:
            labels_m[i] = labels_dict[type]
    if types[i] == "Bone_Marrow":
        labels[i] = 1
    

dff = df.drop(['samples', 'type'], axis = 1)
dff = StandardScaler().fit_transform(dff)
pca = PCA(n_components=16)
pca_dff = pd.DataFrame(data = pca.fit_transform(dff),columns = ['pc'+str(i) for i in range(1,17)] )


def multi_qubit_swap(num_qubits):
    dim = 2**(2*num_qubits)
    matrix = np.zeros((dim,dim))
    for i in range(dim):
        st1 = bin(i)[2:].zfill(2*num_qubits)
        st2 = st1[::-1]
        matrix[int(st1,2), int(st2,2)]=1
    
    return matrix



def distance(A, B):

    wires_num = int(np.ceil(np.log2(len(A))))

    dev = qml.device("default.qubit", wires=2*wires_num+1)
    @qml.qnode(dev)
    def circuit(AA,BB):
        AA_pad, BB_pad  = np.zeros(2**wires_num),np.zeros(2**wires_num)
        AA_pad[:len(AA)], BB_pad[:len(BB)] = AA, BB
        
        feet = np.kron(AA_pad, BB_pad)
        qml.AmplitudeEmbedding(features=feet, wires=range(1,2*wires_num+1), normalize=True)
        qml.Hadamard(wires=0)
        qml.ControlledQubitUnitary(multi_qubit_swap(wires_num), control_wires=[0], wires=range(1, 2*wires_num+1))

        qml.Hadamard(wires=0)
        return qml.probs(wires=0)
    
    x = circuit(A,B)
    x = x[0]
    innerprod = np.sqrt(2*(x-0.5))
    return np.sqrt(2*(1-innerprod))
    # QHACK #


def predict(dataset, labels, new, k):

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data, new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [labels[i] for i in nearest]

    output = k_nearest_classes()

    return 1 if len([i for i in output if i == 1])>len(output)/2 else 0
 

def accuracy(k):
    acc = 0
    for indx,data in enumerate(pca_dff.values):
        dataset_n,labels_n = list(pca_dff.values.copy()),list(labels.copy())
        data_n,label_n = dataset_n.pop(indx),labels_n.pop(indx)
        prediction = predict(dataset_n, labels_n, data_n, k)
        if prediction==label_n:
            acc+=1
    return acc/len(labels)
        
        
#print([accuracy(i) for i in range(3,10)])
#accurracy from k=3 to k=9
#[0.578125, 0.546875, 0.53125, 0.5, 0.5, 0.5625, 0.578125]

# for i in range(3,10):
#     print(f"k={i}, acc: ",accuracy(i))


plt.plot(range(3,10), [accuracy(i) for i in range(3,10)], c='r')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid()
plt.save_fig("plot.svg")