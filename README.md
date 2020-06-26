# AdvAttack 
## Zero Order Optimization Algorithm for Constrained Non-Convex Optimization
This repository contains the code to implements various zero order algorithms used to perform adversarial attacks against black box Deep Neural Networks.

### Requirements
* **Python 3**
* sys
* os
* time
* argparse
* warnings
* json
* datatime
* tqdm
* numpy
* matplotlib
* PIL
* sklearn
* torch
* torchvision


### Code Structure
* papers
* models
* results
    * [optimizer]
    * Summary Results.ipynb
* scripts
    * test.py
    * evaluation.py
* src
    * dataset.py
    * models.py
    * train.py
    * loss.py
    * ZOOptim.py 
    * zeroOptim.py
    * FWOptim.py
* Attack[dataset]_[optimizer].py
* TransferLearning.ipynb


### Description
Papers contains the 4 papers used as references in this project.

Models contains the three models used and their weights: VGG16, InceptionV3 and an ad-hoc network.

Results contains one sub-folder for each Optimizer, each one containing the csv file related to the evaluation of the specified optimizer. It also contains a jupyter notebook used to generate the plots presented in the report.

Scripts contains two python files: *test.py* is used to perform an attack to a single image with the specified Optimizer, *evaluation.py* does the same on a batch of images.

Src contains:
* *dataset.py* is used to retrieve MNIST and CIFAR, to display some informations and eventually to rescale them to be compatible with InceptionV3.
* *models.py* is used to retrieve or create the three networks used (VGG16, InceptionV3 and an Ad-Hoc Network), to train and evaluate them.
* *train.py* is used to train the networks.
* *loss.py* implements the two loss functions used (Mean Squared Erorr and ZOO Loss).
* *[XXX]Optim.py* implement the optimizers described in the papers.

*Attack[dataset]_[optimizer].py* are used to perform a specific attack on a specific dataset and to visualize the generated adversarial image and the loss function.


### Examples
Here we give an example of how to run an evaluation of *Zero Stochastic Conditional Gradient with Inexact Updates* against *VGG16* fine-tuned on *Cifar10*. Only a subset of arguments will be given, full explanation of the arguments can be found inside the scripts. <br>
If the data is not already present in the folder *data* the scripts will automatically create the folder *data* and download the data in there. 

      cd scripts
      evaluation.py --optimizer "inexact" -- data "cifar10" --maximise 0 --epochs 100 --n_gradient 4000 --batch_size 1000 --mu 0.0025 --gamma 1       

**N.B** <br>
1- All the scripts and jupyter notebook should be able to run without a problem in GPU thanks to *cuda* interface. If you don't have *cuda* or you are not able to have it (you don't have an NVIDIA GPU), we strongly suggest you to run everything on *Google Colab*.

2- The models uploaded are only *VGG16* and *MNISTNet* while the *InceptionV3* fine-tuned on *Cifar10* and *MNIST* data are not present. Too fine -tune your *InceptionV3* model one can see *TransferLearning.ipynb* otherwise you can contact us at *francesco.grimaldi.1@studenti.unipd.it* and we will send you the model through Google Drive.
