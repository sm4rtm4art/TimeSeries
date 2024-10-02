import React from 'react';

const Sidebar: React.FC = () => {
  return (
    <aside style={{
      width: '250px',
      height: '100vh',
      backgroundColor: '#f0f0f0',
      padding: '20px',
      boxShadow: '2px 0 5px rgba(0,0,0,0.1)'
    }}>
      <h2>Time Series Forecast</h2>
      <nav>
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          <li><a href="#" style={linkStyle}>Dashboard</a></li>
          <li><a href="#" style={linkStyle}>Forecast</a></li>
          <li><a href="#" style={linkStyle}>Data Upload</a></li>
          <li><a href="#" style={linkStyle}>Settings</a></li>
        </ul>
      </nav>
    </aside>
  );
};

const linkStyle = {
  display: 'block',
  padding: '10px 0',
  color: '#333',
  textDecoration: 'none',
  fontSize: '16px'
};

export default Sidebar;
