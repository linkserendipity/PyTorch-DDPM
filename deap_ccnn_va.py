"""
train a Continuous Convolutional Neural Network (CCNN) on the DEAP dataset for emotion classification.
"""
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
from torcheeg.trainers import ClassificationTrainer
#
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--dir', type=str, required=True, help="data dir")
parser.add_argument('--dir', type=str, default='/media/SSD/lingsen/data/EEG/DEAP/data_preprocessed_python', help="data dir")
parser.add_argument('--save_dir', type=str, default='/media/SSD/lingsen/code/PyTorch-DDPM/save_model/CFG_y_model.pth', help="save model dir")
parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
# parser.add_argument('--timesteps', type=int, default=5000, help="timesteps")
parser.add_argument('--epochs', type=int, default=50, help="epochs")
# parser.add_argument('--image_size', type=int, default=64, help="image_size")
# parser.add_argument('--image_size', type=int, default=128, help="image_size")
parser.add_argument('--gpuid', type=int, default=0, help="GPU ID")
# parser.add_argument('--scale', type=float, default=1.1, help="CFG_scale")

args = parser.parse_args()

data_dir = args.dir 
save_dir = args.save_dir
batch_size = args.batch_size
epochs = args.epochs
gpuid = args.gpuid

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.

os.makedirs('./tmp_out/deap_ccnn_va/log', exist_ok=True)
logger = logging.getLogger('CCNN with the DEAP Dataset')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/deap_ccnn_va/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

###############################################################################
# Set the random number seed in all modules to guarantee the same result when running again.


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

###############################################################################
# Customize Trainer
# -----------------------------------------
# TorchEEG provides a large number of trainers to help complete the training of classification models, generative models and cross-domain methods. Here we choose the simplest classification trainer, inherit the trainer and overload the log function to save the log using our own defined method; other hook functions can also be overloaded to meet special needs.
#


class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


######################################################################
# Building Deep Learning Pipelines Using TorchEEG
# -----------------------------------------
# Step 1: Initialize the Dataset
#
# We use the DEAP dataset supported by TorchEEG. Here, we set an EEG sample to 1 second long and include 128 data points. The baseline signal is 3 seconds long, cut into three, and averaged as the baseline signal for the trial. In offline preprocessing, we divide the EEG signal of every electrode into 4 sub-bands, and calculate the differential entropy on each sub-band as a feature, followed by debaselining and mapping on the grid. Finally, the preprocessed EEG signals are stored in the local IO. In online processing, all EEG signals are converted into Tensors for input into neural networks.
#

dataset = DEAPDataset(io_path=f'./tmp_out/deap_ccnn_va/deap',
                      root_path=data_dir,
                      offline_transform=transforms.Compose([ #@  Band DE 
                          transforms.BandDifferentialEntropy( 
                              sampling_rate=128, apply_to_baseline=True),
                          transforms.BaselineRemoval(),
                          transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT) #@ 转成9*9的Grid
                      ]),
                      online_transform=transforms.ToTensor(),
                      label_transform=transforms.Compose([ #@ valence 二分类
                          transforms.Select(['valence', 'arousal']), #@ list形式输入多个 value
                          transforms.Binary(5.0), 
                          transforms.BinariesToCategory() #@ 添加多标签 [0, 1] 转成0, 1, 2, 3
                      ]),
                      chunk_size=128,
                      baseline_chunk_size=128,
                      num_baseline=3,
                      num_worker=4)

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
# Here, the dataset is divided using 5-fold cross-validation. In the process of division, we group according to the trial index, and every trial takes 4 folds as training samples and 1 fold as test samples. Samples across trials are aggregated to obtain training set and test set.

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path='./tmp_out/deap_ccnn_va/split')

######################################################################
# Step 3: Define the Model and Start Training
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the CCNN model and define the hyperparameters. For example, each EEG sample contains 4-channel features from 4 sub-bands, the grid size is 9 times 9, etc.
# We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # Initialize the model
    print(f'i={i}')
    model = CCNN(in_channels=4, grid_size=(9, 9), num_classes=4, dropout=0.5) #@

    # Initialize the trainer and use the 0-th GPU for training, or set device_ids=[] to use CPU
    trainer = MyClassificationTrainer(model=model,
                                      lr=1e-4,
                                      weight_decay=1e-4,
                                      device_ids=[gpuid])

    # Initialize several batches of training samples and test samples
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    # Do 50 rounds of training
    trainer.fit(train_loader, val_loader, num_epochs=epochs)
    trainer.test(val_loader)
    trainer.save_state_dict(f'./tmp_out/deap_ccnn_va/weight/epoch_{epochs}_{i}.pth')


