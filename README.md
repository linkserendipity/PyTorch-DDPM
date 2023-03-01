# PyTorch-DDPM

DDPM精简后的公式： https://timecat.notion.site/DDPM-b8e2a91927d249fdbcf7c82f2eb6f846

## Classifier-Free DDPM

用 EEG embedding 作为条件输入 生成对应表情的人脸

## 1. Environment

clone the code and create the python environment named `dd`

```bash
git clone https://github.com/linkserendipity/PyTorch-DDPM.git
cd PyTorch-DDPM
conda env create -f environment.yml
conda activate dd
```

## 2. Data preparation

### DEAP EEG dataset

32 subject, 40 one-minute music video, cut into one-second samples, resulting in 32\*40\*60=76800 samples (80% for training, 20% for testing)

Use CCNN to extract EEG samples into 1024-dim vectors and save them to .npy files.

**Put these feature_vector\*_npy.npy and test_feature_vector\*_npy.npy files to your code folder.**


### CK+ facial expression dataset

I choose 4 different facial expression images from CK+ dataset.

The folder of selected images is "~/data/CK+/results/VA"

0:Sad, 1:Angry 2:Calm 3:Happy


|  Label  |  0  |   1   |  2  |   3   |
| :-------: | :---: | :-----: | :----: | :-----: |
| Emotion | Sad | Angry | Calm | Happy |
| Number | 280 |  450  | 327 |  626  |

## 3.Training and testing

**Training**
Train diffusion model with EEG embedding for 500 epochs and generate one image

```bash
python Train_ck_eeg_emb_ca.py --batch_size 12 --timesteps 1000 --epochs 500 --image_size 128 --gpuid 4 --scale 1.6 
```

**Testing**
Generate one image with trained model (**modify the scale from 1.0 to 3.0**)

```bash
python Test_ck_eeg_emb_ca.py --gpuid 4 --scale 1.6
```

## 4.Image Generation

Generate 8*160 = 1280 images for each (gn\*ge)

change **--label** from 0 to 3

```bash
python Test_ck_eeg_emb_ca_gen.py --gpuid 4 --scale 1.6 --gn 8 --ge 160 --label 2
```

## 5.FID

pip install pytroch_fid

Compare the FID score between generated images and training images

```bash
python -m pytorch_fid \
~/data/CK+/results/generated_va_ca_test/128_12_500_1000_1.6 \
~/data/CK+/results/VA \
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
