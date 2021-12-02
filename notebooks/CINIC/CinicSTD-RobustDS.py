import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from inspect import getfullargspec,signature

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
#%%


#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

############################
class CinicResNet(nn.Module):
  def __init__(self, in_channels=3):
    super(CinicResNet, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = models.resnet50(pretrained=False)

    # Change the input layer to take Grayscale image, instead of RGB images.  **************
    # Hence in_channels is set as 1 or 3 respectively
    # original definition of the first layer on the ResNet class
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)

  def forward(self, x):
    return self.model(x)


my_resnet = CinicResNet()

input = torch.randn((16,3,244,244))
output = my_resnet(input)
print(output.shape)

print(my_resnet)
#%%


from torchvision.models import resnet18

from robustness.datasets import CINIC#,RobustDataSet
ds = CINIC('/tmp/cinic')
#replace with the im_adv
# load data from model INSTEAD
from robustness.model_utils import make_and_restore_model
model_adv, _ = make_and_restore_model(arch='resnet50', dataset=ds,
              resume_path= '/home/ashraf.haddad/mlpr/CINIC-50Store-ADV/checkpoint.pt.best') #'/tmp/35daedae-1b39-4941-ad08-8bd6459c1bd8/checkpoint.pt.best')

model_adv.eval()

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in signature(metric_fn).parameters:
        return metric_fn(true_y, pred_y, average="micro")
    else:
        return metric_fn(true_y, pred_y)

def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(torch.cuda.is_available())

model = CinicResNet().to(device)

# params you need to specify:
epochs = 200
NUM_WORKERS = 8
BATCH_SIZE = 128

# Dataloaders
#train_loader, val_loader = #model_adv outputs ds.make_loaders(workers=NUM_WORKERS, batch_size=BATCH_SIZE)

# params you need to specify,
ATTACK_EPS = 1000 #from the https://github.com/MadryLab/robust_representations/blob/master/interpolation.ipynb
ATTACK_STEPSIZE = 0.1
ATTACK_STEPS = 1000
NUM_WORKERS = 8
epochs = 200
BATCH_SIZE = 128
kwargs = {
    'constraint':'2', # use L2-PGD
    'eps': ATTACK_EPS, # L2 radius around original image
    'step_size': ATTACK_STEPSIZE,
    'iterations': ATTACK_STEPS,
    'do_tqdm': True,
    'targeted': True,
    'use_best': False
}
#%%

# Dataloaders
# This code is what a robustness dataset is. We leverage the adv-trained model
# to generate the image which are robust and we construct a tensor dataset
#  and use the dataloader to train a std model on that set.
train_l, val_l = ds.make_loaders(workers=NUM_WORKERS, batch_size=BATCH_SIZE) #train_loader,val_loader
train_enum,val_enum = enumerate(train_l),enumerate(val_l)
train_list = torch.empty(0)
label_t_list = torch.empty(0)
label_v_list = torch.empty(0)
val_list = torch.empty(0)
for _ in range(10000):  #len(train_l)
    _,(im_t,label_t) = next(train_enum) #next(enumerate(train_l))
    _,(im_v,label_v) = next(val_enum)#next(enumerate(val_l))

    #move to CUDA
    im_t,label_t = im_t.to(device),label_t.to(device)
    im_v,label_v = im_v.to(device),label_v.to(device)
    #the model generates our ROBUST dataset\n",
    _, train_robust = model_adv(im_t, label_t, make_adv=True, **kwargs)
    _, val_robust = model_adv(im_v, label_v, make_adv=True, **kwargs)

    #move back to CPU\n",
    im_t,label_t = im_t.to(device='cpu'),label_t.to(device='cpu')
    im_v,label_v = im_v.to(device='cpu'),label_v.to(device='cpu')

    train_list = torch.cat((train_list,train_robust.to(device='cpu')),dim=0)
    val_list = torch.cat((val_list,val_robust.to(device='cpu')),dim=0)
    label_t_list =torch.cat((label_t_list,label_t.to(device='cpu')),dim=0)
    label_v_list =torch.cat((label_v_list,label_v.to(device='cpu')),dim=0)


    #rob_ds = RobustDataSet(img_adv, augmentation)


train_robust_td = TensorDataset(train_list,label_t_list.type(torch.LongTensor))
val_robust_td = TensorDataset(val_list,label_t_list.type(torch.LongTensor))
#train_loader = transforms.Resize((224, 224)).forward(train_robust)\n",
#val_loader =  transforms.Resize((224, 224)).forward(val_robust)\n",

train_loader = DataLoader(train_robust_td,batch_size=128 )#,  num_workers=NUM_WORKERS, pin_memory=True
val_loader = DataLoader(val_robust_td,batch_size=128) #, num_workers=NUM_WORKERS, pin_memory=True)
#%%



# loss function and optimiyer
loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

start_ts = time.time()

losses = []
batches = len(train_loader)
val_batches = len(val_loader)
print(batches,val_batches)


for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    # set model to training
    model.train()

    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)

        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

    # releasing unceseccary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------- VALIDATION  -----------------
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X) # this get's the prediction from the network

            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, accuracy),
                                   (precision_score, recall_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )

    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss/batches) # for plotting learning curve
print(f"Training time: {time.time()-start_ts}s")

#%%

torch.save(model.state_dict(), "./CinicSTD50NetRseT")
model = CinicResNet()
model_state_dict = torch.load("./CinicSTD50NetRseT")
model.load_state_dict(model_state_dict)