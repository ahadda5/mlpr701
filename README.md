A stab at robustness through MadryLab's library. 

The aim is to explore http://madry-lab.ml/ robustness package. 

**Dependencies**
  * Pytorch 3.7 or higher
  * Cudatoolkit 11
  * Scikit-learn 
  * Robustness library by Madry https://github.com/MadryLab/robustness 

**Datasets**
- FashionMnist: This dataset is an interesting and an upgrade from the original MNIST used by Madry's team. The dataset can be cloned directly GitHub where the images are under data/fashion.  git clone https://github.com/zalandoresearch/fashion-mnist.git
- CINIC: This is another interesting and an upgrade from the original CIFAR10 used by Madry's team. Download instructions are [here](https://github.com/BayesWatch/cinic-10)
- COVID XRAY


**Respository Structure**
The following repository is organized as such
* This readme file (markdown file) 
* Two main folders: Robustness and Jupyter Notebooks. The former encompasses all the changes to Robustness library to make it work for the three datasets. those files have to be copied into the conda/python robustness library location.  The second includes all the Notebook and python files needed to generate results specific to our project. 
* Notebooks serve to generate PGD-perturbed examples, PGD-train and standard-train model. Additionally every file is named properly to assist end-users to run them as demos. 
* Every dataset has the following demo/notebook(ipynb) files: 
  -  A standard-training notebook of a ResNet50 against the natural(original set) named:  (DATASET)STD.ipynb
  -  An adv-trained model using the Robustness library  named: Using robustness to ADV train-DATASET.ipynb  (please note that this file is an adaptation of [this](https://github.com/MadryLab/robustness/blob/master/notebooks/Using%20robustness%20as%20a%20library.ipynb) 
  -  A file to generate adversarial accuracy on a std-trained model name: (DATASET)STD-ADV validation.ipynb
  -  A file to generate perturbed(PGD-attacked) examples named: (DATASET)Adv Examples pre-trained ADV.ipynb
  -  A _python_ file to generate a robust dataset(for both Fashion and CINIC) name: (Dataset)STD-RobustDS.py

* FashionMnist and Covid Xray required custom dataloaders. CINIC was already supported by PyTorch. 


