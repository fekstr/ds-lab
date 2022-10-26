import pandas as pd
import os
import sys
import numpy as np

def calculate_stroma_score(path_to_TGCA):
    x = pd.read_excel(path_to_TGCA).to_numpy()
    # print(x.shape)
    x[:,-1] = x[:,-1] / 365.35 # convert 'days to event' to 'years to event'
    x[:,11] = x[:,11] / 10 # convert 'years to birth' to 'decades to birth'
    x = np.append(x, np.zeros((500,1)), axis=1) # add column for HD score

    allCuts = np.array([0.00056, 0.00227, 0.03151, 0.00121, 0.01123, 0.02359, 0.06405, 0.00122, 0.99961]) # Youden cuts
    allWeights = np.array([1.150, 0.015, 5.967, 1.226, 0.488, 3.761, 0.909, 1.154, 0.475]) # HR
    medianTrainingSet = 8.347
    scoreIndices = (np.argwhere(allWeights >= 1)).flatten()

    for i in scoreIndices:
        x[:,-1] = x[:,-1] + (x[:,i+1]>=allCuts[i])*allWeights[i] # +1 retrieve column number in x
    x[:,-1] = (x[:,-1] >= medianTrainingSet)*1

    return x[:,-1]