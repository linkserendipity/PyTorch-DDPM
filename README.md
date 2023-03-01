# PyTorch-DDPM

DDPM精简后的公式： https://timecat.notion.site/DDPM-b8e2a91927d249fdbcf7c82f2eb6f846

## Classifier-Free DDPM

条件控制 DDPM：给定EEG embedding 作为condition 生成对应表情的人脸

## 1. Environment

create the python environment named `dd`
```bash
conda env create -f environment.yml
```

<!-- 最后一行prefix路径要改一下？ -->

## 2. Data preparation

### DEAP EEG dataset

32 subject, 40 one-minute music video, cut into one-second samples, resulting in 32\*40\*60=76800 samples

Use CCNN to extract EEG samples into 1024-dim vectors and save them to .npy files.

**Put these .npy files to your code folder.**

80% for training

/media/SSD/lingsen/code/PyTorch-DDPM/feature_vector0_npy.npy

/media/SSD/lingsen/code/PyTorch-DDPM/feature_vector1_npy.npy

/media/SSD/lingsen/code/PyTorch-DDPM/feature_vector2_npy.npy

/media/SSD/lingsen/code/PyTorch-DDPM/feature_vector3_npy.npy

20% for testing

/media/SSD/lingsen/code/PyTorch-DDPM/test_feature_vector0_npy.npy

/media/SSD/lingsen/code/PyTorch-DDPM/test_feature_vector1_npy.npy

/media/SSD/lingsen/code/PyTorch-DDPM/test_feature_vector2_npy.npy

/media/SSD/lingsen/code/PyTorch-DDPM/test_feature_vector3_npy.npy

### CK+ facial expression dataset

I choose 4 different facial expression images from CK+ dataset.
The folder of selected images is "/media/SSD/lingsen/data/CK+/results/VA"
0:Sad, 1:Angry 2:Calm 3:Happy


|  Label  | 0   |   1   |  2  |   3   |
| :-------: | :-----: | :-----: | :----: | :-----: |
| Emotion | Sad | Angry | Calm | Happy |
| Number | 280 |  450  | 327 |  626  |



## 3.Training and testing

**Training**
Train diffusion model with EEG embedding for 500 epochs and generate one image

```bash
/home/lingsen/miniconda3/envs/dd/bin/python Train_ck_eeg_emb_ca.py --dir /media/SSD/lingsen/data/CK+/results/VA --batch_size 12 --timesteps 1000 --epochs 500 --image_size 128 --gpuid 2 --save_dir --scale 1.8
```

**Testing**
Generate one image with trained model (**modify the scale from 1.0 to 3.0**)

```bash
/home/lingsen/miniconda3/envs/dd/bin/python Test_ck_eeg_emb_ca.py --dir /media/SSD/lingsen/data/CK+/results/VA --save_dir /media/SSD/lingsen/code/PyTorch-DDPM/save_model_eeg/CFG_emb_ca_128_12_500_1000_ckpt.pth --gpuid 4 --scale 2.0
```

## 4.Image Generation

Generate 8*160 = 1280 images for each (gn\*ge)
change **--label** from 0 to 3

```bash
/home/lingsen/miniconda3/envs/dd/bin/python Test_ck_eeg_emb_ca_gen.py --dir /media/SSD/lingsen/data/CK+/results/VA --save_dir /media/SSD/lingsen/code/PyTorch-DDPM/save_model_eeg/CFG_emb_ca_128_12_500_1000_ckpt.pth --scale 1.6 --gpuid 1 --gn 8 --ge 160 --label 2
```

## 5.FID

pip install pytroch_fid

Compare the FID score between generated images and training images

```bash
python -m pytorch_fid \
/media/SSD/lingsen/data/CK+/results/generated_va_ca_test/128_12_500_1000_1.6 \
/media/SSD/lingsen/data/CK+/results/VA \
--device cuda:2
```

## code structure

```
PYTORCH-DDPM
│
└─── Train_ck_eeg_emb_ca.py
│
└─── Test_ck_eeg_emb_ca.py
│
└─── Test_ck_eeg_emb_ca_gen.py
│
└─── models
    │
    └─── ddpm.py
    │
    └─── unet.py
```