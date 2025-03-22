import React, { useState } from 'react';
import axios from 'axios';

interface ForecastGeneratorProps {
  onForecastGenerated: (data: any) => void;
}

const ForecastGenerator: React.FC<ForecastGeneratorProps> = ({ onForecastGenerated }) => {
  const [modelChoice, setModelChoice] = useState('');
  const [modelSize, setModelSize] = useState('small');
  const [forecastHorizon, setForecastHorizon] = useState(10);

  const handleGenerateForecast = async () => {
    try {
      const response = await axios.post('http://localhost:8000/generate_forecast', {
        model_choice: modelChoice,
        model_size: modelSize,
        forecast_horizon: forecastHorizon
      });
      onForecastGenerated(response.data.forecasts);
    } catch (error) {
      console.error('Error generating forecast:', error);
    }
  };

  return (
    <div>
      <h2>Generate Forecast</h2>
      <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
        <option value="">Select Model</option>
        <option value="Chronos">Chronos</option>
        <option value="N-BEATS">N-BEATS</option>
        <option value="Prophet">Prophet</option>
        <option value="TiDE">TiDE</option>
      </select>
      <select value={modelSize} onChange={(e) => setModelSize(e.target.value)}>
        <option value="small">Small</option>
        <option value="medium">Medium</option>
        <option value="large">Large</option>
      </select>
      <input
        type="number"
        value={forecastHorizon}
        onChange={(e) => setForecastHorizon(parseInt(e.target.value))}
        min="1"
      />
      <button onClick={handleGenerateForecast}>Generate Forecast</button>
    </div>
  );
};

export default ForecastGenerator;
