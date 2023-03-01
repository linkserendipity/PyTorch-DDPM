# Generating images using DDPM with EEG embedding as conditioning input
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
from PIL import Image

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
parser.add_argument('--label', type=int, default=0, help="generated emotional images")
parser.add_argument('--gn', type=int, default=16, help="generated image number")
parser.add_argument('--ge', type=int, default=100, help="generated epochs for i")
#@ number of generated images is gn*ge
args = parser.parse_args()

data_dir = args.dir 
save_dir = args.save_dir
batch_size = args.batch_size
timesteps = args.timesteps
image_size = args.image_size
epochs = args.epochs
gpuid = args.gpuid
scale = args.scale
label = args.label
generate_number = args.gn
generate_epochs = args.ge
eeg_dir = args.eeg_dir
eeg_save_dir = args.eeg_save_dir
eeg_batch_size= args.eeg_batch_size
device = "cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu"
message0 = f'eeg_CFG_emb_ca: gpuid={gpuid} scale={scale} batch_size={batch_size} timesteps={timesteps} image_size={image_size} epochs={epochs} device={device}'
logger.info(message0)


############################## load EEG data and CK+ image data
#@ 提取 DEAP 训练集和测试集的 feature 分别保存npy diffusion训练和测试的时候分别读取对应npy 到torch.tensor
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

############################## define model and diffusion function
model = UNetModel(
    in_channels=1, ## RGB images channels = 3 
    model_channels=128, # model_channels=128?
    out_channels=1,
    channel_mult=(1, 2, 2), #?? channel_mult=(1, 2, 2, 4) ？
    attention_resolutions=[],
    label_num=len(dataset.classes) # 4 classes
)
model.load_state_dict(torch.load(save_dir, map_location=device))
model.to(device)
model.eval()
y_o = label*torch.ones([generate_number]).long()
print(f'y_o:{y_o}')
# test feature 生成图片的文件夹
generate_dir = '/media/SSD/lingsen/data/CK+/results/generated_va_ca_test/{}_{}_{}_{}_{}/{}'.format(image_size, batch_size, epochs, timesteps, scale, label)
print(f'generate_dir: {generate_dir}')
os.makedirs(generate_dir, exist_ok=True) 

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule = 'cosine', CFG_scale=scale)
for i in range(0, generate_epochs): #TODO 101
    print(f'i={i}, generate_epochs={generate_epochs}')
    # embed()
    # y = [feature_list[y_o[i].item()][random.randint(0, len(feature_list[y_o[i].item()])-1)] for i in range(len(y_o))] #@ 这还是一个list
    y = [feature_list_test[y_o[i].item()][random.randint(0, len(feature_list_test[y_o[i].item()])-1)] for i in range(len(y_o))] 
    #TODO 变成 test feature了 估计更多不准的图
    y = torch.stack(y, 0) #@@@@ torch.Size([8, 1024])
    y = y.to(device) # 先别放进device 直接是原始的 list 数字进来

    generated_images = gaussian_diffusion.sample(model, y, image_size, batch_size=generate_number, channels=1) 

    imgs = generated_images[-1]
    for j in range(generate_number):
        save_image = os.path.join(generate_dir, f'{i*generate_number+j+1}.png')
        image_array = (imgs[j,0]+1.0) * 255 / 2
        im = Image.fromarray(image_array) 
        im = im.convert('L') # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’ 
        im.save(save_image)
    del generated_images

end_time = round(time.monotonic())
print('Total running time: {}'.format(timedelta(seconds=end_time - start_time))) 
logger.info('Total running time: {}'.format(timedelta(seconds=end_time - start_time)))
message1 = 'EEG_emb_CFG_generation scale={} batch_size={} timesteps={} image_size={} epochs={} generation time={}'.format(scale, batch_size, timesteps, image_size, epochs, timedelta(seconds=end_time - start_time))
logger.critical(message1)
# wx_push(message1)