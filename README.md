# GraphMSI

GraphMSI provides a graph convolutional network-based method to unraveling spatial heterogeneity in mass spectrometry imaging data. 
To flexible apply the GraphMSI into different applications, the GraphMSI can switch its working modes among general, scribble-interactive and knowledge-transfer, enhancing its adaptability to various specimens. 
Developer is Lei Guo from Fuzhou University of China.

# Overflow of GraphMSI model

<div align=center>
<img src="https://github.com/user-attachments/assets/efb1c3d8-8ae4-4719-b650-994db1a63c72" width="500" height="500" /><br/>
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
