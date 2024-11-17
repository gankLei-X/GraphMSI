from trainer import *
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import nor_std
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='GraphMSI for spatial segmentation of mass spectrometry imaging')
parser.add_argument('--input_Matrix',required= True,help = 'path to inputting MSI data matrix')
parser.add_argument('--input_PeakList',required= True,help = 'path to inputting MSI peak list')
parser.add_argument('--input_shape',required= True,type = int, nargs = '+', help='inputting MSI file shape')
parser.add_argument('--n_components',help='Reduced dimension', type = int, default = 20)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()
    data = np.loadtxt(args.input_Matrix,delimiter=',')
    feature = DimensionalityReduction(data, args)
    GraphSlider(feature, args.input_shape)
