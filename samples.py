import numpy as np
import pickle

samples = {}
samples[(1, "Gene A")] = 0
samples[(1, "Gene B")] = 1
samples[(1, "Gene C")] = 1
samples[(1, "Gene D")] = 1
samples[(1, "Gene E")] = 0
samples[(1, "Label")] = True

samples[(2, "Gene A")] = 0
samples[(2, "Gene B")] = 1
samples[(2, "Gene C")] = 1
samples[(2, "Gene D")] = 1
samples[(2, "Gene E")] = 0
samples[(2, "Label")] = True

samples[(3, "Gene A")] = 0
samples[(3, "Gene B")] = 1
samples[(3, "Gene C")] = 1
samples[(3, "Gene D")] = 1
samples[(3, "Gene E")] = 1
samples[(3, "Label")] = True

samples[(4, "Gene A")] = 0
samples[(4, "Gene B")] = 1
samples[(4, "Gene C")] = 0
samples[(4, "Gene D")] = 1
samples[(4, "Gene E")] = 0
samples[(4  , "Label")] = False

samples[(5, "Gene A")] = 1
samples[(5, "Gene B")] = 0
samples[(5, "Gene C")] = 1
samples[(5, "Gene D")] = 0
samples[(5, "Gene E")] = 0
samples[(5, "Label")] = False

samples[(6, "Gene A")] = 0
samples[(6, "Gene B")] = 1
samples[(6, "Gene C")] = 0
samples[(6, "Gene D")] = 0
samples[(6, "Gene E")] = 0
samples[(6, "Label")] = False


with open('samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
        
