# DDPM trained on CK+ dataset with EEG embedding as conditioning input
# EEG embedding: using CCNN to extract feature vectors from DEAP dataset 
# Cross attention for label_embedding
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
import random 

from models.unet import UNetModel
from models.ddpm import GaussianDiffusion
# from wechat_push import wx_push

############################## main
## args definition
start_time = round(time.monotonic())
logger.info("Cross attention !!! EEG feature npy load done! Begin Classifier-Free diffusion!")
writer2 = SummaryWriter()

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
parser.add_argument('--gpuid', type=int, default=2, help="GPU ID")
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
#@ 提取 DEAP 训练集和测试集的 feature 分别保存npy diffusion训练和测试的时候分别读取对应npy 到torch.tensor
#@ feature_list 后面用于把label 换成 随机采样 label对应的 eeg feature 
feature_list=[torch.from_numpy(np.load(f'feature_vector{i}_npy.npy')) for i in range(0, 4)] 
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
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 四类情绪图片 0->Sad 1->Angry 2->Calm 3->Happy
#@ LVLA=0 LVHA=1 HVLA=2 HVHA=3
#@ Sad    Angry  Calm Happy 
# 其中 Calm 是把所有类别的第一张图提出来 Happy是把Happy类别的后10张图提出来 再手动删掉一些看起来不Happy的图 


############################## define model and diffusion function
model = UNetModel(
    in_channels=1, ## RGB images channels = 3 
    model_channels=128, # model_channels=128?
    out_channels=1,
    channel_mult=(1, 2, 2), #?? channel_mult=(1, 2, 2, 4) ？
    attention_resolutions=[],
    label_num=len(dataset.classes) # 4 classes
)
model.to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule = 'cosine') # 改成'cosine'
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


############################## Training stage
# 保存diffusion model的文件夹
os.makedirs('./save_model_eeg/', exist_ok=True)
for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):
        # images.shape = torch.Size([16, 1, 128, 128])
        batch_size1 = images.shape[0] # 最后一个 step 的 images[0]<= 一开始给定的 batch_size(32)
        optimizer.zero_grad()

        images = images.to(device)
        #NOTE 这里把labels 代进feature_list[labels[i]] random sample
        label_emb = [feature_list[labels[i].item()][random.randint(0, len(feature_list[labels[i].item()])-1)] for i in range(len(labels))] # 这还是一个list labels最后一个batch长度可能小于 batch_size
        label_emb = torch.stack(label_emb, 0) # torch.Size([32, 1024])
        label_emb = label_emb.to(device) 

        # sample t uniformally for every example in the batch  随机生成 (batch_size1,) 形状的 [0,timesteps] 范围内采样的t 
        t = torch.randint(0, timesteps, (batch_size1,), device=device).long()
        loss = gaussian_diffusion.train_losses(model, images, t, label_emb) 
        if step % 100 == 0:
            print("epoch:{}, step:{}, Loss:{}".format(epoch, step, loss.item()))
        loss.backward()
        optimizer.step()
        niter = epoch * len(train_loader) + step
        writer2.add_scalar('Train/Loss', loss.item(), niter)
    if (epoch+1) % 100 == 0:  # 每100个epoch保存.pth
        print(f"epoch:{epoch}")
        save_dir = os.path.join("/media/SSD/lingsen/code/PyTorch-DDPM/save_model_eeg", f"CFG_emb_ca_{image_size}_{batch_size}_{epoch+1}_{timesteps}_ckpt.pth")
        logger.info(save_dir)
        torch.save(model.state_dict(), save_dir)
writer2.close()


############################## testing stage 每类生成4张图片
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).to(device) 
yy=y # 生成的图片标题选取对应的label
print('y:{}'.format(y)) 
y = [feature_list_test[y[i].item()][random.randint(0, len(feature_list_test[y[i].item()])-1)] for i in range(len(y))] #@ 从EEG测试集的特征向量里随机采样 
y = torch.stack(y, 0) #@@@@ torch.Size([32, 1024])
y = y.to(device) 
generated_images = gaussian_diffusion.sample(model, y, image_size, batch_size=16, channels=1) 

# generate new images
os.makedirs('./output_eeg_ca/', exist_ok=True) 
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(4, 4)
imgs = generated_images[-1].reshape(4, 4, image_size, image_size)
for n_row in range(4):
    for n_col in range(4):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
        plt.title(f"condition: {yy[n_row*4+n_col]+1}") # 改成+1 对应emotion label
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
print('Total running time: {}'.format(timedelta(seconds=end_time - start_time))) 
logger.info('Total running time: {}'.format(timedelta(seconds=end_time - start_time)))
message1 = 'EEG_emb_CFG_ca scale={} batch_size={} timesteps={} image_size={} epochs={} training time={}'.format(scale, batch_size, timesteps, image_size, epochs, timedelta(seconds=end_time - start_time))
logger.critical(message1)
# wx_push(message1)