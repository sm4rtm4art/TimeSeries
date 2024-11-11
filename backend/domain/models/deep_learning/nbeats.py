from darts.models import NBEATSModel
from backend.core.interfaces.model import TimeSeriesPredictor
from darts import TimeSeries

class NBEATSPredictor(TimeSeriesPredictor):
    def __init__(self):
        self.model_name = "N-BEATS"
        self.model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            n_epochs=100,
            verbose=True
        )
        self.is_trained = False

    def train(self, data: TimeSeries) -> None:
        self.model.fit(data)
        self.is_trained = True

    def predict(self, horizon: int) -> TimeSeries:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} model is not trained yet!")
        return self.model.predict(n=horizon) 