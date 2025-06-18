# 🌡️ NOAATemp_Analyzer

**NOAATemp_Analyzer** is a lightweight, human-in-the-loop Streamlit app designed for exploring global land and ocean temperature anomalies published by NOAA. It enables intuitive data exploration, anomaly spotting, and volatility analysis — **not to predict the future**, but to help users reason more effectively with real-world climate data.

---

## 📌 Purpose

This tool supports users tackling questions like:

> _Will the monthly global temperature anomaly exceed 1.50°C in 2025?_  
> _What might July 2025 look like compared to past Julys?_

Rather than guessing blindly, **NOAATemp_Analyzer** offers interactive visuals and metrics to support informed speculation or forecasting.

---

## ⚙️ Features

- 📈 **Interactive time series viewer** for NOAA monthly anomaly data  
- 🔍 **Volatility Analysis** — Quantify stability or turbulence in temperature trends  
- 📉 **Spike & Dip Detector** — Identify months with unusually high or low anomalies  
- 🎛️ **Custom time range selection** for focused exploration  
- 🧠 **Exploratory, not predictive** — Designed to enhance human judgment

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** – UI framework
- **Pandas** – Data handling
- **NumPy** – Statistical calculations
- **Plotly** or **Matplotlib** – Visualizations
- **CSV input** – Global anomalies sourced from [NOAA's Global Monitoring](https://www.ncei.noaa.gov/)
