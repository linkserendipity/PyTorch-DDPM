# Testing DDPM model with EEG embedding as conditioning input
import os
import math
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import argparse
from log import logger
from tensorboardX import SummaryWriter
import numpy as np
import random #@ random.randint(0,len(feature_list[i]))
import logging
from IPython import embed
from models.unet import UNetModel
from models.ddpm import GaussianDiffusion
# from wechat_push import wx_push

############################## main
## args definition
start_time = round(time.monotonic())
logger.info("Testing! EEG feature  load done! Begin image generation!")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/media/SSD/lingsen/data/CK+/results/VA', help="data dir")
parser.add_argument('--eeg_dir', type=str, default='/media/SSD/lingsen/data/EEG/DEAP/data_preprocessed_python', help="EEG data dir")
parser.add_argument('--save_dir', type=str, default='/media/SSD/lingsen/code/PyTorch-DDPM/save_model_eeg/CFG_emb_ca_128_12_500_1000_ckpt.pth', help="save model dir")
parser.add_argument('--eeg_save_dir', type=str, default='/media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/weight', help="eeg model dir")
parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument('--eeg_batch_size', type=int, default=256, help="EEG CCNN batch_size")
parser.add_argument('--timesteps', type=int, default=5000, help="timesteps")
parser.add_argument('--epochs', type=int, default=200, help="epochs")
parser.add_argument('--image_size', type=int, default=128, help="image_size")
parser.add_argument('--gpuid', type=int, default=3, help="GPU ID")
parser.add_argument('--scale', type=float, default=1.5, help="CFG_scale: the scale of classifier-free guidance")
args = parser.parse_args()

data_dir = args.dir 
save_dir = args.save_dir
batch_size = args.batch_size
timesteps = args.timesteps
image_size = args.image_size
epochs = args.epochs
gpuid = args.gpuid
scale = args.scale
eeg_dir = args.eeg_dir
eeg_save_dir = args.eeg_save_dir
eeg_batch_size= args.eeg_batch_size
device = "cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu"
message0 = f'eeg_CFG_emb_ca: gpuid={gpuid} scale={scale} batch_size={batch_size} timesteps={timesteps} image_size={image_size} epochs={epochs} device={device}'
logger.info(message0)


############################## load EEG data and CK+ image data
#TODO 提取 DEAP 训练集和测试集的 feature 分别保存到npy里面去 diffusion训练和测试的时候分别读取对应的feature npy 到tensor
#@ feature_list 后面用于把label 换成 随机采样 label对应的 eeg feature 
# feature_list=[torch.from_numpy(np.load(f'feature_vector{i}_npy.npy')) for i in range(0, 4)] 
feature_list_test=[torch.from_numpy(np.load(f'test_feature_vector{i}_npy.npy')) for i in range(0, 4)]

## 对images resize 裁剪 norm
transform = transforms.Compose([
    transforms.Grayscale(1), #@ 3通道转成单通道
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
dataset = ImageFolder(data_dir, transform = transform) 
print("dataset.class_to_idx: ", dataset.class_to_idx)
############################## define model and diffusion function
model = UNetModel(
    in_channels=1, ## RGB images channels = 3 
    model_channels=128, # model_channels=128?
    out_channels=1,
    channel_mult=(1, 2, 2), #?? channel_mult=(1, 2, 2, 4) ？
    attention_resolutions=[],
    label_num=len(dataset.classes) # 4 classes
)
model.load_state_dict(torch.load(save_dir, map_location=device)) #@@ loaded_model
model.to(device)
model.eval()

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule = 'cosine') #TODO 改成'cosine'


############################## testing stage 每类生成4张图片
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).to(device) 
yy=y # 生成的图片标题选取对应的label
print('y:{}'.format(y)) 
y = [feature_list_test[y[i].item()][random.randint(0, len(feature_list_test[y[i].item()])-1)] for i in range(len(y))] 
y = torch.stack(y, 0) #@@@@ torch.Size([32, 1024])
y = y.to(device) 
generated_images = gaussian_diffusion.sample(model, y, image_size, batch_size=16, channels=1) 

# generate new images
# os.makedirs('./output_eeg_ca/', exist_ok=True) 
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(4, 4)
imgs = generated_images[-1].reshape(4, 4, image_size, image_size)
for n_row in range(4):
    for n_col in range(4):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
        plt.title(f"condition: {yy[n_row*4+n_col]+1}") #@ 改成+1 对应emotion label
plt.savefig("output_eeg_ca/aa_emb_ca_{}_{}_{}_{}_scale{}_output.png".format(image_size, batch_size, epochs, timesteps, scale))
plt.close(fig)

# show the denoise steps
fig1 = plt.figure(figsize=(12, 12), constrained_layout=True)
gs1 = fig1.add_gridspec(16, 16)

for n_row in range(16):
    for n_col in range(16):
        f_ax = fig1.add_subplot(gs1[n_row, n_col])
        t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
        img = generated_images[t_idx][n_row].reshape(image_size, image_size)
        f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
        plt.title(f"{yy[n_row]+1}")
plt.savefig("output_eeg_ca/aa_emb_ca_{}_{}_{}_{}_scale{}_reverse.png".format(image_size, batch_size, epochs, timesteps, scale))
plt.close(fig1)

end_time = round(time.monotonic())
logger.info('Total running time: {}'.format(timedelta(seconds=end_time - start_time)))
message1 = 'Test EEG_emb_CFG_ca scale={} batch_size={} timesteps={} image_size={} epochs={} testing time={}'.format(scale, batch_size, timesteps, image_size, epochs, timedelta(seconds=end_time - start_time))
logger.critical(message1)
# wx_push(message1)