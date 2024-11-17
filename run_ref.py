from trainer import *
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='GraphMSI for spatial segmentation of mass spectrometry imaging')
parser.add_argument('--input_Matrix',required= True,help = 'path to inputting MSI data matrix')
parser.add_argument('--input_PMatrix',required= True,help = 'path to inputting MSI data matrix')
parser.add_argument('--input_PeakList',required= True,help = 'path to inputting MSI peak list')
parser.add_argument('--input_shape',required= True,type = int, nargs = '+', help='inputting MSI file shape')
parser.add_argument('--input_Pshape',required= True,type = int, nargs = '+', help='inputting MSI file shape')

parser.add_argument('--mode',
                    help = 'General mode for default segmentation, Scribble-interactive mode for enhance segmentation, Transfer-knowledge mode for faster segmentation',
                    default= 'General')
parser.add_argument('--n_components',help='Reduced dimension', type = int, default = 20)
parser.add_argument('--use_scribble',help='use scribbles', metavar='1 or 0', default=0, type=int)
parser.add_argument('--use_reference',help='use reference', metavar='1 or 0', default=1, type=int)
parser.add_argument('--output_file', default='output/',help='output file name')

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    setup_seed(0)

    args = parser.parse_args()
    data = np.loadtxt(args.input_Matrix,delimiter=',')
    feature = DimensionalityReduction(data, args)

    graph = GraphConstruction(feature, args.input_shape,0.216)
    im_target = FeatureClustering(feature, graph, args)
    np.savetxt(args.output_file + '.txt', im_target)

    dataP = np.loadtxt(args.input_PMatrix,delimiter=',')
    im_target = Predicting(dataP,args)
    np.savetxt(args.output_file + 'Pre.txt', im_target)
