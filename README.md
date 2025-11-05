# Solar Production Prediction System

A machine learning pipeline for predicting solar energy production using weather data and temporal features. Implements ensemble methods combining LightGBM, XGBoost, and Random Forest for accurate forecasting.

## Overview

This project predicts hourly solar panel output based on meteorological conditions. The methodology is informed by comparative analysis research on regression algorithms for solar energy prediction.

**Key Features:**
- Time-series aware data processing
- Advanced feature engineering (lag variables, rolling statistics, cyclical encoding)
- Ensemble learning with hyperparameter optimization
- Comprehensive evaluation and visualization

## Dataset

Hourly solar production measurements with weather conditions:
- SystemProduction (target)
- Radiation
- Sunshine
- AirTemperature
- WindSpeed
- Date-Hour(NMT)

## Methodology

### Feature Engineering
- **Temporal features**: Hour, day, month, season, cyclical encoding
- **Lag features**: Historical values at 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 168h
- **Rolling statistics**: Mean, max, min over 3h, 6h, 12h, 24h, 48h windows
- **Interactions**: Radiation × Sunshine, Radiation × Temperature

### Models
1. **LightGBM**: Fast gradient boosting with optimized parameters
2. **XGBoost**: Regularized gradient boosting
3. **Random Forest**: Ensemble of 300 decision trees
4. **Voting Ensemble**: Weighted combination (LightGBM: 2, XGBoost: 2, RF: 1)

### Evaluation
- TimeSeriesSplit cross-validation (3 folds)
- Metrics: R² Score, RMSE, MAE
- 80/20 time-based train/test split

## Results

| Metric | Value |
|--------|-------|
| R² Score | 0.5286 |

The model explains 53% of variance in solar production, which is solid performance for renewable energy forecasting given weather variability and atmospheric effects not captured in basic meteorological data.

## Installation
```bash
git clone https://github.com/yourusername/solar-production-prediction.git
cd solar-production-prediction
pip install -r requirements.txt
```

**Requirements:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0

## Project Structure
```
├── Solar.csv                  # Input dataset
├── solar_prediction.py        # Main pipeline
├── best_model.pkl            # Trained model
├── scaler.pkl                # Feature scaler
├── model_results.csv         # Performance metrics
└── requirements.txt          # Dependencies
```

## Visualizations

The pipeline generates:
- Time series plots of production
- Hourly production patterns
- Correlation heatmap
- Scatter plots of key relationships
- Model performance comparison
- Actual vs predicted plots
- Residual analysis
- Feature importance rankings

## Key Insights

**Most Important Features:**
- Radiation intensity (primary driver)
- Sunshine duration
- Historical production values
- Rolling radiation statistics
- Hour of day (cyclical)

**Model Performance:**
- Ensemble outperforms individual models
- LightGBM offers best speed/accuracy tradeoff
- XGBoost provides robust generalization

## Limitations

- Limited meteorological features (no cloud cover, humidity, pressure)
- No panel-specific data (temperature, degradation, soiling)
- Static model without online learning
- Does not provide prediction uncertainty estimates

## Future Improvements

- Add numerical weather prediction data
- Incorporate satellite imagery for cloud cover
- Implement LSTM/Transformer architectures
- Add prediction intervals with quantile regression
- Build real-time API for deployment
- Integrate automated retraining pipeline

## References

**Primary Reference:**  
Solar energy prediction through machine learning models: A comparative analysis of regressor algorithms - PubMed Central (PMC)

**Additional:**
1. Pedregosa et al. (2011) - Scikit-learn: Machine Learning in Python, JMLR 12
2. Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System, KDD
3. Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NIPS

## License

MIT License
