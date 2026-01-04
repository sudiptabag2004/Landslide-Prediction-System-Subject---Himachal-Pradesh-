# 🏔️ Landslide Susceptibility Assessment System

A **research-oriented, inference-only web application** for assessing landslide susceptibility using a **pre-trained CatBoost machine learning model**, designed for **Himachal Pradesh, India**.

This system integrates **user-defined environmental scenarios**, **threshold-based decision control**, **historical landslide context**, and **local explainability (SHAP)** within a clean, publication-safe interface.

---

## 📌 Project Overview

Landslides are complex, rainfall-triggered geohazards influenced by terrain, land cover, and antecedent hydrological conditions.

This project provides a **decision-support dashboard** that allows researchers and analysts to:

- Evaluate landslide susceptibility probabilities
- Explore recall–false alert trade-offs via adjustable thresholds
- Inspect historical landslide occurrence near a selected location
- Interpret model predictions using feature-level explainability

> ⚠️ **Important**  
> This system is **not an operational early warning system**.  
> It is intended **solely for research, sensitivity analysis, and scientific demonstration**.

---

## 🎯 Key Features

### 1. Machine Learning–Based Susceptibility Prediction
- **Model:** CatBoost Classifier  
- **Output:** Probability of landslide occurrence, `P(landslide)`  
- **Inference-only:** No online learning or real-time ingestion  

### 2. Threshold-Controlled Decision Logic
- Adjustable classification threshold (0.1–0.8)
- Displays:
  - Recall (sensitivity)
  - Alert rate
- Enables controlled exploration of detection vs. false-alarm trade-offs

### 3. Historical Landslide Context Integration
- Uses an offline CSV-based landslide inventory
- Computes:
  - Number of past landslides within 20 km
  - Nearest event distance
  - Most recent event year
- Historical events are **contextual only** and **independent of model predictions**

### 4. Spatial Visualization
- Interactive OpenStreetMap-based view
- Displays:
  - Selected location
  - Nearby historical landslide events (if available)

### 5. Explainable AI (XAI)
- Local feature attribution using **SHAP**
- Visualizes each feature’s contribution to the final prediction
- Fallback importance-based explanation if SHAP is unavailable

### 6. Research-Safe UI Design
- Progressive result reveal (alert → metrics → history → explanation)
- Subtle transitions and animations
- No distracting or non-scientific UI elements

---

## 🧪 Model Inputs

The model expects the following features:

| Feature   | Description                         |
|----------|-------------------------------------|
| rain_1d  | 1-day rainfall (mm)                 |
| rain_3d  | 3-day cumulative rainfall (mm)      |
| rain_7d  | 7-day cumulative rainfall (mm)      |
| slope    | Slope angle (degrees)               |
| lulc     | Land use / land cover (encoded)     |

Land-use categories are internally encoded for model compatibility.

---

## 🗺️ Study Area

- **Region:** Himachal Pradesh, India  
- **Characteristics:**
  - Himalayan terrain with steep slopes
  - Monsoon-driven extreme rainfall
  - Diverse land-use patterns
  - High landslide density

---

## 🧠 System Architecture (High-Level)

