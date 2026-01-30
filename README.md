# BCI_Final
## 📋 Project Overview

This project implements **MotorFormer**, a Transformer-based deep learning model for EEG-based motor imagery classification. The model is designed to decode left-hand vs right-hand movement intentions from EEG signals, with applications in Brain-Computer Interfaces (BCI) and neurorehabilitation.

## Set up
### Dataset
You can download the dataset from [Dataset](https://doi.org/10.5524/100295) and place it in the EEG_data folder
### Installation


'''bash
# 1. Clone repository
git clone https://github.com/yiwenchen617/MotorFormer.git
cd MotorFormer

# 2. Create virtual environment
conda create -n motorformer python=3.8 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cudatoolkit=11.8 -c pytorch -c conda-forge -y
conda activate motorformer
'''
