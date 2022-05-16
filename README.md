# Supporting code for: Resolving complex cartilage structures in developmental biology via deep learning-based automatic segmentation of X-ray computed microtomography images

This repository contains supporting code for our work dealing with deep learning-based segmentation of the cartilaginous nasal capsule in micro-CT images of mouse embryos.

# Abstract
The complex shape of embryonic cartilage represents a true challenge for phenotyping and basic understanding of skeletal development. X-ray computed microtomography (μCT) enables inspecting relevant tissues in all three dimensions; however, most 3D models are still created by manual segmentation, which is a time-consuming and tedious task. In this work, we utilised a convolutional neural network (CNN) to automatically segment the most complex cartilaginous system represented by the developing nasal capsule. The main challenges of this task stem from the large size of the image data (over a thousand pixels in each dimension) and a relatively small training database, including genetically modified mouse embryos, where the phenotype of the analysed structures differs from the norm. We propose a CNN-based segmentation model optimised for the large image size that we trained using a unique manually annotated database. The segmentation model was able to segment the cartilaginous nasal capsule with a median accuracy of 84.44% (Dice coefficient). The time necessary for segmentation of new samples shortened from approximately 8 h needed for manual segmentation to mere 130 s per sample. This will greatly accelerate the throughput of μCT analysis of cartilaginous skeletal elements in animal models of developmental diseases.

# Project structure
```
mouse-cartilage-segment
│   README.md
│
└───code
│   │   environment.yml - contains information to install the conda environment
│   │   example_train_model.py - a python script with a basic training loop to train the cartilage segmentation CNN proposed in our work
│   │   model_residual_selu_deeplysup.py - the proposed model architecture
│   │   notebook_segment_cartilage.ipynb - Jupyter notebook showing the application of one of our trained models on new data and interactive visualisation of the results
│
└───data
│   └───Sample_slices
│   │   └───Images - a random sample of 50 images from our database of micro-CT scans of 17 day old embryos
│   │   │   │   slice_0000.tif
│   │   │   │   .
│   │   │   │   .
│   │   │   │   .
│   │   │   │   slice_0049.tif
│   │   └───Ground_truth - corresponding manually segmented masks of the nasal capsule cartilage
│   │   │   │   mask_0000.tif
│   │   │   │   .
│   │   │   │   .
│   │   │   │   .
│   │   │   │   mask_0049.tif   
│   └───Trained_models - contains models in h5 file format trained on the entire database as presented in the manuscript for each of the cross-validation runs
│   │   │   │   val0_residual_selu_deeplysup.h5
│   │   │   │   val1_residual_selu_deeplysup.h5
│   │   │   │   val2_residual_selu_deeplysup.h5
│   │   │   │   val3_residual_selu_deeplysup.h5
│   │   │   │   val4_residual_selu_deeplysup.h5
│   │   │   │   val5_residual_selu_deeplysup.h5
│   │   │   │   val6_residual_selu_deeplysup.h5
│
└───results
│   └───Example_trained_model - here the results of the example training script are stored
│   │   └───example_trained_model
│   │   │   │   best_training_model.h5 - model with the best average training loss in epoch
│   │   │   │   best_validation_model.h5 - model with the best average validation loss in epoch
│   │   │   │   latest_model.h5 - model from the last training epoch
│   │   │   │   example_trained_model.txt - training hyperparameters stored as txt file
│   │   │   │   training_loss.txt - pickled average training loss in each epoch
│   │   │   │   validation_loss.txt - pickled average validation loss in each epoch
│   │
│   └───Predicted_masks - here the results of the prediction using the already trained models is stored 
│   │   │   predicted_0000.tif
│   │   │   .
│   │   │   .
│   │   │   .
│   │   │   predicted_0049.tif
│   │   │   results.txt - here the Dice segmentation accuracy coefficient for each segmented slice is stored
```
# Requirements
The yaml file containing the dependencies to run this code can be found in the ``code/environment.yml`` file. We strongly recommend using the [Anaconda](https://www.anaconda.com/) package manager. To install the required dependencies using Anaconda, run the ``conda env create -f environment.yml`` in your Conda terminal and activate the environment by running the ``conda activate cartilage-seg-environment`` command. The CNN training script can be run as ``python example_train_model.py``. The application of the trained models on the data can be run inside of the Jupyter Notebook environment, which offers the possibility to interactively visualise the segmentation results.
The code was tested on a Windows workstation equipped with an nVidia Quadro P5000 GPU with 16 GB of graphical memory, 512 GB of RAM and an Intel® Xeon® Gold 6248R CPU.

# Reference
If you use any part of our work in your research, please consider citing the following publication:

Matula, J. et al. Resolving complex cartilage structures in developmental biology via deep learning-based automatic segmentation of X-ray computed microtomography images. Scientific Reports; 10.1038/s41598-022-12329-8 (2022).
