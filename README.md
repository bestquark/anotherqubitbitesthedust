# AnotherQubitBitesTheDust - QHack22

## Project: QGenes

### Description of the project

Developments in microarray technologies have revolutionized life sciences by giving us the capability to simultaneously measure thousands of gene expression values. This presents us with the golden opportunity to use quantum computers to process these vast amounts of information to train machine learning models, which we can use to accurately predict the possibility of developing genetic disorders from gene expressions of particular individuals. 

In our project, we use limited freely available data of only 64 gene expression samples obtained from different leukemia patients and implement a k-NN algorithm that predicts Bone Marrow Leukemia with 80% accuracy. This involves using amplitude embedding to represent N gene expression features in log_2(n) qubits to represent the vector from one sample and use swap test to calculate its distance from other samples. Then we can assign the label of the sample to be the same as the majority labels of its nearest k neighbors. The advantage here is that we need to use exponentially fewer qubits than bits to represent our samples in the vector space. We can further improve our algorithm by using more labels for different diseases and exploring other classification algorithms such as SVMs and Quantum Kernels.
#### Data source: [Kaggle - Bruno Grisci](https://www.kaggle.com/brunogrisci/leukemia-gene-expression-cumida)

### Source code: 
[GitHub Repo]([http://github.com](https://github.com/BestQuark/anotherqubitbitesthedust/blob/main/kNN%20leukemia/knn-qml.py))

### List of Open Hackathon challenges:
- Bio-QML Challenge
- Quantum entrepreneur challenge
