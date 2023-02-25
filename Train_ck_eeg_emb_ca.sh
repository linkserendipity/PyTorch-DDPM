python Train_ck_eeg_emb_ca.py --dir /media/SSD/lingsen/data/CK+/results/VA \
--batch_size 14 \
--timesteps 2000 \
--epochs 500 \
--image_size 128 \
--gpuid 2 \
--save_dir /media/SSD/lingsen/code/PyTorch-DDPM/save_mode_eeg/CFG_eeg_emb_ca_model_12_600_2000_s1.6.pth \
--scale 1.8 \
--eeg_dir /media/SSD/lingsen/data/EEG/DEAP/data_preprocessed_python \
--eeg_save_dir /media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/weight \
--eeg_batch_size 256 