# Time Series Forecasting Tool for

A comprehensive time series forecasting platform that enables consultants to upload, clean, analyze, visualize, and forecast data with minimal technical knowledge. The platform follows clean code principles, SOLID design, and provides an intuitive interface with powerful backend capabilities.

## Project Status

![In Development](Images/istockphoto-527660774-612x612.jpg)

**Current Phase**: Prototype development with focus on data pipeline implementation

- ✅ Basic Streamlit interface implemented
- ✅ Development environment with UV configured
- ✅ Type safety foundation with MyPy
- 🔄 In Progress: Robust data processing pipeline
- 🔄 In Progress: Code quality improvements

_Image source: [iStock](https://media.istockphoto.com/id/527660774/vector/under-construction-industrial-sign.jpg?s=612x612&w=0&k=20&c=3U2TR5u_Drl4B5HBRc13wHD32nZe38UhlB6hzkj93U0=)_

## Key Features

1. **Data Upload & Validation**

   - Multi-format support (CSV, Excel, JSON)
   - Intelligent error detection and reporting
   - Schema inference with Pydantic validation
   - Robust data pipeline for preprocessing

2. **Data Exploration & Cleaning**

   - Automated column type detection
   - Interactive data preview and profiling
   - Outlier detection and handling
   - NaN handling strategies
   - Time series-specific transformations

3. **Model Training & Selection**

   - Multiple time series models:
     - N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)
     - TIDE (Temporal Importance-Guided Denoising Encoder)
     - Prophet (Facebook's time series forecasting tool)
     - TimeMixer (A novel approach combining temporal convolutions and attention)
     - TFT (Temporal Fusion Transformers)
     - ARIMA (Autoregressive Integrated Moving Average)
   - Automated model selection based on data characteristics
   - Custom model configuration
   - Experiment tracking with MLflow/Neptune

4. **Forecasting & Analysis**

   - Interactive forecast visualization
   - Confidence intervals and uncertainty quantification
   - Feature importance analysis
   - What-if scenario testing

5. **Reporting & Export**
   - Customizable PDF reports
   - Excel/CSV exports with metadata
   - Interactive dashboard sharing

## Technical Stack

### Frontend

- **Current**: Streamlit (for rapid prototyping)
- **Planned**: React with TypeScript, Tailwind CSS, Redux Toolkit, React Query
- **Visualization**: Plotly.js, D3.js

### Backend

- **API Framework**: FastAPI with Pydantic
- **Forecasting Libraries**: Darts, Prophet
- **Data Processing**: Polars, NumPy, scikit-learn
- **Testing**: Pytest, Hypothesis
- **Documentation**: OpenAPI/Swagger
- **Type Checking**: MyPy with strict typing
- **Data Version Control**: DVC (planned)
- **Configuration**: Dynaconf (planned)

### Infrastructure

- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring & Tracking**:
  - System Monitoring: Prometheus, Grafana (planned)
  - ML Experiment Tracking: MLflow/Neptune (planned)

## Development Roadmap

### Prototype Phase (Current)

- Streamlit-based interface
- Core data processing pipeline
- Basic model integration
- Data validation and cleaning

### MVP Phase (Next)

- Enhanced time series processing
- Comprehensive model evaluation
- Improved Streamlit UI
- Basic reporting capabilities
- Documentation and testing

### Production Phase (Future)

- React frontend migration
- Advanced modeling features
- Comprehensive monitoring
- Deployment automation
- Enterprise security features

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/timeseries.git
cd timeseries

# Set up Python environment with UV
python -m venv .timeseries
source .timeseries/bin/activate  # On Windows: .timeseries\Scripts\activate
pip install uv
uv pip install -r requirements.txt

# Run the Streamlit app (for development)
cd backend/app
PYTHONPATH=../.. streamlit run streamlit.py  # On Windows: set PYTHONPATH=..\.. && streamlit run streamlit.py
```

### Running Tests

```bash
# From project root
pytest tests/
```

## Project Structure

```
timeseries/
├── backend/
│   ├── api/                     # API endpoints
│   │   ├── routes/              # API route definitions
│   │   └── schemas/             # API request/response schemas
│   ├── app/                     # Main application code
│   ├── application/             # Application layer
│   │   └── services/            # Application-specific services
│   ├── Cleaning/                # Data cleaning modules
│   │   └── models/              # Cleaning models and algorithms
│   ├── config/                  # Configuration settings
│   ├── core/                    # Core functionality
│   │   ├── config/              # Core configuration
│   │   └── interfaces/          # Core interfaces/protocols
│   ├── data/                    # Data processing modules
│   ├── domain/                  # Domain layer (business logic)
│   │   ├── models/              # Domain models
│   │   │   ├── boosting/        # Gradient boosting models
│   │   │   ├── deep_learning/   # Neural network models
│   │   │   ├── experimental/    # Experimental model implementations
│   │   │   └── statistical/     # Statistical forecasting models
│   │   └── services/            # Domain services
│   ├── infrastructure/          # Infrastructure layer
│   │   ├── results/             # Results storage and management
│   │   ├── storage/             # Data storage components
│   │   └── ui/                  # UI integration components
│   ├── tests/                   # Test suite
│   └── utils/                   # Utility functions
├── frontend/                    # React frontend (planned)
│   ├── public/                  # Static assets
│   └── src/                     # Source code
│       ├── assets/              # Frontend assets
│       └── components/          # React components
├── Images/                      # Project images
├── k8s/                         # Kubernetes configuration
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── Anomaly detection/       # Anomaly detection experiments
│   ├── Darts/                   # Darts library experiments
│   ├── m3/                      # M3 competition datasets
│   ├── Other/                   # Miscellaneous notebooks
│   └── preprocessing/           # Data preprocessing experiments
└── darts_logs/                  # Model training logs
```

The project follows a clean architecture approach with distinct layers:

- **API Layer**: Handles HTTP requests and responses
- **Application Layer**: Orchestrates use cases and workflows
- **Domain Layer**: Contains business logic and model implementations
- **Infrastructure Layer**: Provides implementation details for external interactions

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on code style, testing requirements, and process.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Contact

For any inquiries, please open an issue in the GitHub repository.
