A stab at robustness through MadryLab's library. 

The aim is to explore https://github.com/MadryLab/ robustness package. 

FashionMnist was added to the dataloaders, to be adv-trained and compared to the std-trained model accuracy. 

CINIC was used as well, but was already supported by PyTorch. 

Xray COVID dataset required its custom loaders . 

For each sets, the following 3 metrics were computed
* Standard training of a model against the natural(original set) and then validated against an PGD adversary
* PGD-trained model validated against the natural and the adversarial dataset
* Standard model trained on Adversarial dataset, then validated aginst natural and adv dataset. 

Team members- 
Mugariya Farooq
Ashraf Haddad
Sharim Jamal
