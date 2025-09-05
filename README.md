# 💰 CryptoWallet — Cryptocurrency Price Forecasting (LSTM + Sentiment)

🚀 A deep learning-powered **cryptocurrency price prediction** web app built with **Flask**, **LSTM (TensorFlow/Keras)**, and **VADER Sentiment Analysis**.  

Predict **future prices**, analyze **market sentiment**, and visualize **crypto trends** — all in a clean, interactive interface.  

---

## 🔗 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black?logo=flask)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?logo=tensorflow)  
![NLP](https://img.shields.io/badge/VADER-Sentiment_Analysis-green)  
![Yahoo Finance](https://img.shields.io/badge/YFinance-API-yellow)  
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-red?logo=databricks)  

---

## ✨ Features

✅ **Real-time Data** — Fetches historical prices via **Yahoo Finance**  
✅ **Deep Learning Model** — LSTM trained on crypto time-series  
✅ **Sentiment Analysis** — Market mood detection via **VADER**  
✅ **Interactive UI** — Flask + dynamic charts & plots  
✅ **User Authentication** — Secure login/signup system  
✅ **Email Alerts** — Price notifications using SMTP  
✅ **Attractive Dashboard** — Rich visuals, easy navigation  

---

## 🖼️ Screenshots  

### 🏠 Home Page
<img width="1882" alt="Home" src="https://github.com/user-attachments/assets/1643a2fb-1b18-41d8-acc9-d6e597b60146" />

### 🔍 Input Page
<img width="1862" alt="Input" src="https://github.com/user-attachments/assets/44a47c12-6359-4908-ae61-bfc916e2bd1d" />

### 📊 Prediction Results
<img width="1417" alt="Results" src="https://github.com/user-attachments/assets/14c11bda-7049-4e14-a3e5-4553fb8545b8" />
<img width="1520" alt="Graph" src="https://github.com/user-attachments/assets/dbe86ae8-e7b0-46e1-bfde-844f98ee70b0" />
<img width="1454" alt="Output" src="https://github.com/user-attachments/assets/927f4f1c-4393-4673-997c-c0ff948a2f0e" />

### 📈 Analysis Dashboard
<img width="1599" alt="Analysis1" src="https://github.com/user-attachments/assets/6796402b-3196-4909-b487-bf2c96f739d3" />
<img width="1603" alt="Analysis2" src="https://github.com/user-attachments/assets/3bf3a914-1ed5-40e9-b97f-8bd1459e48f4" />
<img width="1577" alt="Analysis3" src="https://github.com/user-attachments/assets/f99a5a9d-6862-45b5-8724-966c027fa785" />

---

## ⚡ How It Works

1. **Fetch Data** — Collects real-time & historical prices with `yfinance`  
2. **Preprocess Data** — Normalizes time-series for LSTM input  
3. **Train LSTM** — Predicts next-step crypto prices  
4. **Sentiment Analysis** — Scrapes news headlines, applies **VADER**  
5. **Final Prediction** — Combines LSTM forecast + sentiment trend  

---


## License
Copyright (c) 2025 [Varun M C].  
All rights reserved.  
This project is licensed under the [License](LICENSE) terms.

##  Deployment

**[Crptowallet]([https://crytpowallet-1.onrender.com])**

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/yvarun/CryptoWallet.git
cd CryptoWallet

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py


