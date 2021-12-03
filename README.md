A stab at robustness through MadryLab's library. 

The aim is to explore http://madry-lab.ml/ robustness package. 

**Dependencies**
    * Pytorch 3.7 or higher
    * Cudatoolkit 11
    * Scikit-learn 
    * Robustness library by Madry https://github.com/MadryLab/robustness 


**Team members**
- Mugariya Farooq
- Ashraf Haddad
- Sharim Jamal

**Datasets**
- FashionMnist: This dataset is an interesting and an upgrade from the original MNIST used by Madry's team. The dataset can be cloned directly GitHub where the images are under data/fashion.  git clone git@github.com:zalandoresearch/fashion-mnist.git
- CINIC: This is another interesting and an upgrade from the original CIFAR10 used by Madry's team. 

The following repository is organized as such
* The following readme file (markdown file) 
* Two main folders: Robustness and Jupyter Notebooks. The former encompasses all the changes to Robustness library to make it work for the three datasets. The second includes all the Notebook and python files needed to generate results specific to our project. 
* Notebooks serve to generate PGD-perturbed examples, PGD-train and standard train model. Additionally every file is named properly to assist, end-users to run them as demos. 
•Provide a demo file available with sample inputs and outputs.
•Provide instructions on downloading data from publicly available links (for the datasets used in the project)
•If a project is built on an existing code-base, it must be clearly credited and differences should be explicitly stated in the readme file.

FashionMnist was added to the dataloaders, to be adv-trained and compared to the std-trained model accuracy. 

CINIC was used as well. For both sets, the following 3 metrics were computed
* Standard training of a model against the natural(original set) and then validated against an PGD adversary
* PGD-trained model validated against the natural and the adversarial dataset
* Standard model trained on Adversarial dataset, then validated aginst natural and adv dataset. 
