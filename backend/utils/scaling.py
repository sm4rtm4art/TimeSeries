from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def scale_data(data: TimeSeries) -> tuple[TimeSeries, Scaler]:
    """
    Scale the input TimeSeries data using Darts' Scaler.
    
    :param data: Input TimeSeries data
    :return: Tuple of scaled TimeSeries and the Scaler object
    """
    scaler = Scaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def inverse_scale(data: TimeSeries, scaler: Scaler) -> TimeSeries:
    """
    Inverse scale the TimeSeries data using the provided Scaler.
    
    :param data: Scaled TimeSeries data
    :param scaler: Scaler object used for initial scaling
    :return: Inverse scaled TimeSeries
    """
    return scaler.inverse_transform(data)
