import React, { useState } from 'react';
import { ThemeProvider, createGlobalStyle } from 'styled-components';

const lightTheme = {
  background: '#ffffff',
  text: '#000000',
};

const darkTheme = {
  background: '#222222',
  text: '#ffffff',
};

const GlobalStyle = createGlobalStyle`
  body {
    background-color: ${props => props.theme.background};
    color: ${props => props.theme.text};
  }
`;

function App() {
  const [theme, setTheme] = useState(lightTheme);
  const [forecastData, setForecastData] = useState(null);

  const toggleTheme = () => {
    setTheme(theme === lightTheme ? darkTheme : lightTheme);
  };

  const handleForecastGenerated = (data) => {
    setForecastData(data);
  };

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <div style={{ display: 'flex' }}>
        <Sidebar onThemeToggle={toggleTheme} />
        <main style={{ flex: 1, padding: '20px' }}>
          <h1>Time Series Forecast</h1>
          <ForecastGenerator onForecastGenerated={handleForecastGenerated} />
          <D3Chart data={forecastData} />
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;
