# Darts

### Makes time series forecasting easy

Darts is a Python library for user-friendly forecasting and anomaly detection on time series. It contains a variety of models, from classics such as ARIMA to deep neural networks. The forecasting models can all be used in the same way, using fit() and predict() functions, similar to scikit-learn. The library also makes it easy to backtest models, combine the predictions of several models, and take external data into account. Darts supports both univariate and multivariate time series and models. The ML-based models can be trained on potentially large datasets containing multiple time series, and some of the models offer a rich support for probabilistic forecasting.

Darts also offers extensive anomaly detection capabilities. For instance, it is trivial to apply PyOD models on time series to obtain anomaly scores, or to wrap any of Darts forecasting or filtering models to obtain fully fledged anomaly detection models.

### Installation

```bash
pip install darts
```

### Example

```python
from darts import TimeSeries

# Create a TimeSeries object
series = TimeSeries.from_series(pd.Series([1, 2, 3, 4, 5]))

# Create a model
model = ExponentialSmoothing()

# Fit the model
model.fit(series)

# Predict the next 10 steps
prediction = model.predict(10)
```

### Resources

[Darts Webside](https://unit8co.github.io/darts/)

[Darts_Blogpost](https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878)
