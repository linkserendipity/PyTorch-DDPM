# torcheeg
#! test only
"""CCNN with the DEAP Dataset
======================================
In this case, we introduce how to use TorchEEG to train a Continuous Convolutional Neural Network (CCNN) on the DEAP dataset for emotion classification.
"""

import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms as eeg_transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.trainers import ClassificationTrainer
from torcheeg.models import CCNN
from models.ccnn import CCNN_emb #@@@@@@@@@@@@@

#
import argparse
from IPython import embed
import random

parser = argparse.ArgumentParser()

# parser.add_argument('--dir', type=str, required=True, help="data dir")
parser.add_argument('--dir', type=str, default='/media/SSD/lingsen/data/EEG/DEAP/data_preprocessed_python', help="data dir")
parser.add_argument('--eeg_save_dir', type=str, default='/media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/weight/0.pth', help="save model dir")
parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
# parser.add_argument('--timesteps', type=int, default=5000, help="timesteps")
parser.add_argument('--epochs', type=int, default=80, help="epochs")
# parser.add_argument('--image_size', type=int, default=64, help="image_size")
# parser.add_argument('--image_size', type=int, default=128, help="image_size")
parser.add_argument('--gpuid', type=int, default=0, help="GPU ID")
# parser.add_argument('--scale', type=float, default=1.1, help="CFG_scale")

args = parser.parse_args()
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ main 开始训练

data_dir = args.dir 
save_dir = args.eeg_save_dir
batch_size = args.batch_size
epochs = args.epochs
gpuid = args.gpuid

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.

os.makedirs('./tmp_out/deap_ccnn_va_test/log', exist_ok=True)
logger = logging.getLogger('test CCNN model with the DEAP Dataset')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/deap_ccnn_va_test/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

###############################################################################
# Set the random number seed in all modules to guarantee the same result when running again.

#TODO 这个设置了之后 下一次运行代码 每次随机生成的数是不一样的，但是和上一次对应的位置的数是一样的 所以得到diffusion里面取消随机性
def seed_everything(seed):
    random.seed(seed) #@ 这个为什么也要固定？？
    np.random.seed(seed) #TODO 这个要不得吧？
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
                      offline_transform=eeg_transforms.Compose([ #@  Band DE 
                          eeg_transforms.BandDifferentialEntropy( 
                              sampling_rate=128, apply_to_baseline=True),
                          eeg_transforms.BaselineRemoval(),
                          eeg_transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT) #@ 转成9*9的Grid
                      ]),
                      online_transform=eeg_transforms.ToTensor(),
                      label_transform=eeg_transforms.Compose([ #@ valence 二分类
                          eeg_transforms.Select(['valence', 'arousal']), #@ list形式输入多个 value
                          eeg_transforms.Binary(5.0), 
                          eeg_transforms.BinariesToCategory() #@ 添加多标签 [0, 1] 转成0, 1, 2, 3
                      ]),
                      chunk_size=128,
                      baseline_chunk_size=128,
                      num_baseline=3,
                      num_worker=4)

# .. note::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using 5-fold cross-validation. In the process of division, we group according to the trial index, and every trial takes 4 folds as training samples and 1 fold as test samples. Samples across trials are aggregated to obtain training set and test set.
#

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path='./tmp_out/deap_ccnn_va/split')

######################################################################
# Step 3: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the CCNN model and define the hyperparameters. For example, each EEG sample contains 4-channel features from 4 sub-bands, the grid size is 9 times 9, etc.
#
# We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # Initialize the model
    if i != 3:
        print(f'i={i}')
        continue
    print(f'i=3? {i}')
    # loaded_model = CCNN(in_channels=4, grid_size=(9, 9), num_classes=4, dropout=0.5) #@
    loaded_model = CCNN_emb(in_channels=4, grid_size=(9, 9), num_classes=4, dropout=0.5) #@
    save_dir_i = os.path.join(save_dir, f'{i}.pth')
    # save_dir_i = os.path.join(save_dir, f'epoch_80_{i}.pth')
    print(save_dir_i)
    # /media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/weight/epoch_80_0.pth
    # loaded_model.load_state_dict(torch.load(save_dir_i).keys())
    #@ torch.load(save_dir_i).keys() = dict_keys(['model'])
    #@ torch.load(save_dir_i)['model'].keys()=odict_keys(['conv1.1.weight', 'conv1.1.bias', 'conv2.1.weight', 'conv2.1.bias', 'conv3.1.weight', 'conv3.1.bias', 'conv4.1.weight', 'conv4.1.bias', 'lin1.0.weight', 'lin1.0.bias', 'lin2.weight', 'lin2.bias'])
    #? x['model']['conv1.1.weight'].shape = torch.Size([64, 4, 4, 4])
    # Initialize the trainer and use the 0-th GPU for training, or set device_ids=[] to use CPU
    trainer = MyClassificationTrainer(model=loaded_model,
                                      lr=1e-4,
                                      weight_decay=1e-4,
                                      device_ids=[gpuid])
    trainer.load_state_dict(load_path=save_dir_i)

    # Initialize several batches of training samples and test samples
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
    print(f'len(train_loader)={len(train_loader)}') # 240
    print(f'len(val_loader)={len(val_loader)}') # 60
    # Do 50 rounds of training
    # trainer.fit(train_loader, val_loader, num_epochs=epochs)
    # trainer.test(val_loader)
    # trainer.feature_vector(val_loader)

    feature_vector, feature_label = trainer.feature_vector(train_loader) #! train_loader 不是 train_dataset!!!!
    print(feature_vector.shape) #@ torch.Size([15360, 1024]) 61440, 1024
    print(feature_label.shape) #@ torch.Size([15360]) 61440

    index0 = torch.where(feature_label==0)
    index1 = torch.where(feature_label==1)
    index2 = torch.where(feature_label==2)
    index3 = torch.where(feature_label==3)
    feature_vector0 = feature_vector[index0]
    feature_vector1 = feature_vector[index1]
    feature_vector2 = feature_vector[index2]
    feature_vector3 = feature_vector[index3]
    #TODO 不同的数量不能把这四组张量组合成一个大的张量？ 到diffusion里面label对应的feature_vector提出哪一个？
    #! 添加到list里面不就行了 feature_list[0]

    feature_list=[feature_vector0, feature_vector1, feature_vector2, feature_vector3]
    print(feature_list[0].shape)
    print(feature_list[1].shape)
    print(feature_list[2].shape)
    print(feature_list[3].shape)

    #TODO random 都被固定了还怎么取随机数？？？

    # print(feature_vector0.shape)
    # print(feature_vector1.shape)
    # print(feature_vector2.shape)
    # print(feature_vector3.shape)
    embed()
    #NOTE #* 刚好 训练集每个label对应的样本数量是测试集同一个label对应样本数量的4倍
    #@ torch.Size([12480, 1024]) 
    # torch.Size([14208, 1024])
    # torch.Size([12768, 1024])
    # torch.Size([21984, 1024])
    #i=0 3210 3553 3192 5496 说明HVHL占的比例最高 
    #@ i=1 2 3 4 都是一样的数量分布 但是 train test划分不一样 所以对应的feature不一样
    # print(feature_label.shape)

    # embed()
    # trainer.save_state_dict(f'./tmp_out/deap_ccnn_va/weight/{i}.pth')


