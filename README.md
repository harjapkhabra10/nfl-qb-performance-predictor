# NFL QB Performance Predictor

This project explores NFL quarterback performance prediction using historical game data, rolling offensive/defensive trends, and matchup-based features. After getting into fantasy football, I became interested in how volatile QB performance can be from week to week and wanted to see how accurately future production could be modeled using ML.

The project generates future QB projections for NFL Weeks 13–18 of the 2025 season and visualizes predictions through an interactive Streamlit dashboard.

Built with Python, XGBoost, LightGBM, and Streamlit using NFL data from `nflreadpy`.

---

## Features

- QB passing yard, completion, and attempt projections
- Rolling offensive and defensive feature engineering
- Matchup-based future game predictions
- Interactive Streamlit dashboard for weekly QB analysis
- Multiple regression models including XGBoost, Random Forest, and LightGBM

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Streamlit
- nflreadpy

---

## Project Structure

- `nflqbprediction.ipynb`
  - Used for exploratory analysis, feature engineering, and model experimentation

- `modelpipeline.py`
  - Contains the finalized ML pipeline for preprocessing, training, and generating predictions

- `nflstreamlit.py`
  - Streamlit dashboard used to interactively view QB projections and weekly predictions

---

## Running the Project

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm streamlit nflreadpy "polars<1.13"
```

Run the Streamlit app:

```bash
streamlit run nflstreamlit.py
```