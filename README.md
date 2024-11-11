# GraphMSI

GraphMSI provides a graph convolutional network-based method to unraveling spatial heterogeneity in mass spectrometry imaging data. 
To flexible apply the GraphMSI into different applications, the GraphMSI can switch its working modes among general, scribble-interactive and knowledge-transfer, enhancing its adaptability to various specimens. 
Developer is Lei Guo from Fuzhou University of China.

# Overflow of GraphMSI model

<div align=center>
<img src="https://github.com/user-attachments/assets/efb1c3d8-8ae4-4719-b650-994db1a63c72" width="650" height="650" /><br/>
</div>

__Overflow of the proposed GraphMSI for MSI segmentation__. (A) GraphMSI is designed to integrate the metabolite profiles and spatial location for each spot to generate the segmentation results for spatial heterogeneity analysis. 
(B) GraphMSI takes as inputs the MSI data that includes the metabolite profiles and spatial location. Latent embedding data is obtained using parametric-UMAP to preserve the informative features from the metabolite profiles. 
Then, the spatial neighborhood graph is constructed based on the spot coordinates. Both of them are inputted into the two GCN layers and a classifier to obtain the spatial segmentation result. 
(C) The GraphMSI model is trained using the multi-task learning. Particularly, the scribble-interactive mode incorporates the partial or coarse biological contexts to fine-tuning the model to achieve enhanced results. 
(D) The GraphMSI with knowledge-transfer mode trains on MSI dataset from slice-1 and then directly applies the trained model to perform segmentation on unseen MSI dataset from slice-2, which are adjacent slices, without the need for re-training.

# Requirement

    python == 3.5, 3.6 or 3.7
    
    pytorch == 1.8.2
    
    opencv == 4.5.3
    
    matplotlib == 2.2.2

    numpy >= 1.8.0
    
    umap == 0.5.1
    
# Quickly start

## Input
The input is the preprocessed MSI data with two-dimensional shape [X*Y,P], where X and Y represent the pixel numbers of horizontal and vertical coordinates of MSI data, and P represents the number of ions.

## Run GraphMSI model

cd to the GraphMSI fold

If you want to perfrom iSegMSI for unsupervised segmentation, taking mouse kidney data as an example, run:

    python run.py --input_Matrix data/kidney_132_205.csv --input_shape 132 205 --n_components 20 --use_scribble 0 --input_PeakList kidney_peak.csv --output_file output

If you want to perfrom iSegMSI for interactive segmentation, taking fetus mouse data as an example, run:

    python run.py -input_file .../data/fetus_mouse.txt --input_shape 202 107 1237 --DR_mode umap --n_components 3 --use_scribble 1 -- input_scribble .../data/fetus_mouse_scribble.txt --output_file output.txt

If you want to perfrom iSegMSI for hyperparameter search, taking fetus mouse data as an example, run:

    python hyperparameter_earch.py -input_file .../data/fetus_mouse.txt --input_shape 202 107 1237 --DR_mode umap --n_components 3 --use_scribble 0 --output_file output
    
# Contact

Please contact me if you have any help: gl5121405@gmail.com
