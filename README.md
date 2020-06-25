# AdvAttack 
###Zero Order Optimization Algorithm for Constrained Non-Convex Optimization


## Code Structure
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
