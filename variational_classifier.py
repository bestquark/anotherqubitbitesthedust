import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")

print( data.head())