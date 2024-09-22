import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import D3Chart from './components/D3Chart';
import ForecastGenerator from './components/ForecastGenerator';

function App() {
  const [forecastData, setForecastData] = useState(null);

  const handleForecastGenerated = (data) => {
    setForecastData(data);
  };

  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <main style={{ flex: 1, padding: '20px' }}>
        <h1>Time Series Forecast</h1>
        <ForecastGenerator onForecastGenerated={handleForecastGenerated} />
        <D3Chart data={forecastData} />
      </main>
    </div>
  );
}

export default App;