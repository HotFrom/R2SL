# R2SL: Region-Aware Dual Latent State Learning for QoS Prediction

This repository contains the official implementation of **R2SL**, a dual-latent state framework for accurate and scalable Quality of Service (QoS) prediction in large-scale service networks. The method is described in our paper:

> **Ziliang Wang, Xiaohong Zhang, Ze Shi Li, Meng Yan.**  
> *A Region-Aware Dual Latent State Mining Framework for Service Recommendation in Large-Scale Service Networks.*  


## 🔍 Introduction

R2SL addresses two key challenges in QoS prediction:

- **Data sparsity**: by modeling regional latent states (city-level and AS-level) to aggregate distributed QoS logs;
- **Label imbalance**: by introducing a distribution-aware loss function called **Smooth Huber Loss**.

R2SL integrates two key components:

- **Latent State Mining** using EM + Gradient Descent over region-based records.
- **Sparse Mixture-of-Experts (MoE)** neural network for multi-task prediction (Response Time & Throughput).

## 🏗️ Architecture

📊 Dataset
We use the WS-Dream dataset containing QoS logs for 339 users and 5825 services.
Please place your processed .txt files (e.g., 0.02_rt_lda5_train.txt) into the data/ folder.

🚀 Usage
1. Pre-train Regional Latent States
python latent_factor_mining.py
This performs EM + GD to extract latent states for physical and virtual regions, and saves them to new training/test files.

2. Train the R2SL Deep Model

python r2sl_model.py
This trains the R2SL neural network with dual-task outputs: Response Time (RT) and Throughput (TP).

📈 Results
Our method achieves state-of-the-art performance on WS-Dream, reducing MAE by up to 26.4% for response time and 18.8% for throughput compared to strong baselines like FRLN, NCRL, and QoSGNN.


