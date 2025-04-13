# Time Series Forecasting Tool

## Project Vision

A comprehensive time series forecasting platform that enables to upload, clean, analyze, visualize, and forecast data with minimal technical knowledge. The platform follows clean code principles, SOLID design, and provides an intuitive interface with powerful backend capabilities.

## Current Priorities & Next Steps

**Immediate Focus (Phase 1 / Week 0-1):**

- **Fix Critical Code Issues:** Address the syntax error in `tsmixer_model.py` and the security concern in `main.py`.
- **Establish Code Quality Practices:** Finalize critical file fixes, define coding standards (`CONTRIBUTING.md`), and refine the CI pipeline (`ci_backend.yaml`).
- **Complete Core Setup:** Refine project structure, create the backend Docker container, and set up configuration management (Dynaconf).

**Next Major Goal:** Achieve **MVP Phase** status by delivering a robust data pipeline, enhanced time series capabilities, and core infrastructure.

## Next Major Milestone: MVP Phase

**Goal**: Deliver a valuable product with essential features for real-world use

**Key Requirements for MVP Status**:

1.  **Robust Data Pipeline**:

    - Comprehensive data validation and cleaning
    - Automatic column type detection and classification
    - Flexible data transformation options
    - NaN handling and outlier detection
    - Data sampling and preview functionality

2.  **Enhanced Time Series Capabilities**:

    - Feature engineering specific to time series
    - Time series decomposition
    - Stationarity tests and transformations
    - Multiple model types with basic configuration

3.  **Improved User Experience (via Backend API)**:

    - Intuitive data exploration views (driven by backend)
    - Model comparison visualizations (driven by backend)
    - Basic report generation
    - Documentation and user guides

4.  **Core Infrastructure**:
    - Basic monitoring setup
    - Simple data versioning
    - Docker containerization
    - Initial CI/CD pipeline

## Project Roadmap

_(This roadmap outlines the planned steps towards MVP and beyond. Refer to "Current Priorities" above for immediate tasks.)_

### Phase 1: Foundation, Code Quality & Initial Setup (Approx. 1-2 Weeks)

#### Week 0/1: Immediate Tasks & Setup Consolidation

- [ ] **Fix Critical Code Issues:**
  - [ ] Fix Syntax Error in `tsmixer_model.py` (Bandit reported issue)
  - [ ] Address Security Issue in `main.py` (Binding to `0.0.0.0` - investigate, document, or configure)
- [ ] **Establish Code Quality Practices:**
  - [ ] Fix remaining critical file issues (e.g., `streamlit.py` docstrings/types - adapt for backend focus)
  - [ ] Define coding standards in `CONTRIBUTING.md`
  - Note: Use `git commit --no-verify` pragmatically for WIP during quality ramp-up if needed.
- [x] Set up development environment with UV for dependency management
- [x] Create type-safe session state management for Streamlit (Review if still needed for backend testing/tools)
- [x] Configure MyPy for static type checking with pre-commit hooks
- [x] Fix package configuration in `pyproject.toml`
- [ ] Refine project structure following clean architecture principles
- [ ] Implement initial CI/CD pipeline with GitHub Actions (incl. linting, basic tests) - Review `ci_backend.yaml`
- [ ] Create Docker container for backend service
- [ ] Set up Dynaconf for configuration management across environments

#### Week 2: Test Infrastructure & Core Components

- [ ] Expand test coverage for core components (aim for >85% on new/critical logic)
- [ ] Add integration tests for initial API endpoints and data handling
- [ ] Implement performance benchmarks for critical operations (placeholder for future)
- [ ] Set up documentation generation from docstrings (e.g., Sphinx, MkDocs)
- [ ] Create comprehensive README with setup and usage instructions
- [ ] Expand test coverage for core forecasting components
- [ ] Add integration tests with realistic data scenarios
- [ ] Implement performance benchmarks for critical operations
- [ ] Set up documentation generation from docstrings
- [ ] Create comprehensive README with setup instructions
- [ ] Define coding standards in CONTRIBUTING.md
- [x ] Set up development environment with UV for dependency management
- [ ] Refine existing project structure following clean architecture principles
- [x] Implement CI/CD pipeline with proper testing
- [x] Create Docker containers for both frontend and backend
- [ ] Design database schema for storing user data, models, and results
- [ ] Set up MLflow/Neptune for experiment tracking
- [ ] Implement DVC for dataset and model versioning (initial setup)

### Phase 2: Data Processing & Validation (Approx. 3 weeks)

#### Week 3: Data Pipeline Foundation - Ingestion & Validation

- [ ] **Design Core Data Pipeline Components:**
  - [ ] Create `interfaces/protocols.py` (or similar) for data pipeline abstractions (DataReader, DataValidator, DataTransformer, DataWriter) with type hints and docstrings.
- [ ] **Implement Basic Data Reader:**
  - [ ] Create `data_reader.py` (or similar) for CSV, Excel (optional: JSON) with error handling and tests.
- [ ] **Implement Data Validator:**
  - [ ] Create `data_validator.py` (or similar) using Pydantic for schema/type validation, basic constraints (nulls), with tests.
- [ ] Develop data upload endpoints (FastAPI) with validation using the new components.
- [ ] Add error handling with proper type checking for API layer.
- [ ] Design and implement initial data storage strategy (e.g., local files, object storage).
- [ ] Create initial data pipeline orchestration (linking reader, validator).
- [ ] Add unit tests for data validation components within the pipeline.
- [ ] Initialize DVC repositories for dataset versioning.
- [ ] Consider scalability layer design (Polars/PySpark) - placeholder.

#### Week 4: Data Exploration & Column Classification

- [ ] Develop algorithms/logic for automatic column type detection (time, numeric, categorical) within the pipeline.
- [ ] Build API endpoints to support column selection and target definition.
- [ ] Implement data sampling and preview functionality via API.
- [ ] Create backend logic for initial visualizations of raw data (API to return plottable data).
- [ ] Develop data profiling report generation (backend process).
- [ ] Optimize Polars dataframe operations for large datasets.
- [ ] Add integration tests for data exploration API workflows.

#### Week 5: Data Cleaning & Transformation

- [ ] **Begin Data Transformer Implementation:**
  - [ ] Create `data_transformer.py` (or similar) with initial NaN handling (mean, median, etc.), type conversion, simple scaling/normalization. Add tests.
- [ ] Implement outlier detection algorithms with Polars.
- [ ] Build out feature transformation tools (scaling, normalization).
- [ ] Develop custom transformation options (API design).
- [ ] Create API endpoints for data comparison views (before/after cleaning).
- [ ] Implement transformation caching for performance (design).
- [ ] Add unit tests for all transformation operations.

### Phase 3: Time Series Processing & Model Training (Approx. 3 weeks)

#### Week 6: Time Series-Specific Processing

- [ ] Implement time series decomposition (trend, seasonality, residuals).
- [ ] Build frequency detection and conversion logic.
- [ ] Create feature engineering specific to time series (lags, rolling windows).
- [ ] Implement stationarity tests and transformations.
- [ ] Create pipeline for integrating Polars with Darts/Chronos.
- [ ] Add typed interfaces for all time series operations.
- [ ] Implement feature store concept (design/placeholder).

#### Week 7: Initial Model Training

- [ ] Integrate Darts models (statistical, ML, deep learning).
- [ ] Begin Chronos integration (experimental).
- [ ] Build training pipeline with proper cross-validation (using Darts/Sklearn utilities).
- [ ] Implement API for hyperparameter configuration.
- [ ] Create model storage and versioning with MLflow.
- [ ] Track all experiments with Neptune/MLflow.
- [ ] Add type checking for model interfaces.
- [ ] Implement transfer learning capabilities (leveraging Darts pre-trained models where applicable).

#### Week 8: Model Evaluation

- [ ] Develop backend logic for comprehensive metrics calculation.
- [ ] Implement API endpoints for backtesting visualization data.
- [ ] Create API endpoints for forecast vs. actual comparison data.
- [ ] Build backend logic for model comparison tools.
- [ ] Implement API endpoints for confidence interval visualization data.
- [ ] Create experiment comparison UI support via MLflow/Neptune API/data.
- [ ] Add integration tests for complete model workflows (API driven).

### Phase 4: Advanced Features & Optimization (Approx. 3 weeks)

#### Week 9: Advanced Model Features

- [ ] Feature importance calculation and API endpoints for visualization data.
- [ ] Sensitivity analysis tools (backend logic).
- [ ] Ensemble model creation (backend logic).
- [ ] Retrain functionality for selected models (API endpoint).
- [ ] Advanced model tuning options (API design).
- [ ] Implement distributed training support (placeholder/design).
- [ ] Add performance benchmarks for model training.
- [ ] Set up model deployment workflow (initial design with Kubeflow in mind).
- [ ] Develop synthetic data generation using TimeGAN and DeepEcho (experimental).

#### Week 10: Report Generation & Export

- [ ] Design backend logic for PDF report templates.
- [ ] Implement Excel/CSV export with forecasts via API.
- [ ] Create backend for customizable dashboards (API data sources).
- [ ] Build sharing and collaboration features (backend support).
- [ ] Implement scheduled report generation (backend task scheduling).
- [ ] Add ML experiment reporting from MLflow/Neptune (API access).
- [ ] Add type-safe interfaces for all export functions.

#### Week 11: LLM Integration & Advanced Analytics

- [ ] Integrate LLM (Claude/OpenAI) for data interpretation (backend service).
- [ ] Build natural language query interface (API design).
- [ ] Implement automated insights generation (backend process).
- [ ] Create what-if scenario analysis (backend logic and API).
- [ ] Develop executive summary generation (backend process).
- [ ] Add secure authentication for API access (integrate fastapi-users/jwt).
- [ ] Implement end-to-end tests for complete user journeys (API focused).
- [ ] Extend with external data integration capabilities (API design).

### Phase 5: Deployment & Production Readiness (Approx. 2 weeks)

#### Week 12: Cloud Deployment

- [ ] Set up Kubernetes cluster with proper resource allocation.
- [ ] Create Helm charts for deployment management.
- [ ] Implement auto-scaling based on usage patterns.
- [ ] Configure proper security settings and access controls.
- [ ] Set up monitoring and alerting with Prometheus and Grafana.
- [ ] Configure Prometheus Alertmanager for notification routing.
- [ ] Implement backup and recovery strategies.
- [ ] Deploy MLflow/Neptune services.
- [ ] Add deployment documentation.
- [ ] Configure Evidently for model monitoring in production.
- [ ] Set up blue/green deployment for models with Kubeflow.

#### Week 13: Performance Optimization & Final Testing

- [ ] Perform load testing and optimization (API endpoints).
- [ ] Implement caching strategies (FastAPI Cache).
- [ ] Conduct final integration testing with frontend (if available).
- [ ] Fix bugs and implement feedback.
- [ ] Prepare API documentation and usage guides.
- [ ] Optimize Polars operations for production.
- [ ] Create production readiness checklist.
- [ ] Finalize automated deployment pipeline.

## Technical Stack

### Frontend (Separate Repository)

- **Note**: The frontend (React/TypeScript) is developed and maintained in a separate repository. This backend project focuses on providing the necessary APIs.

### Backend

- **API Framework**: FastAPI with Pydantic for validation
- **Forecasting Libraries**: Darts, Chronos
- **Data Processing**: Polars, NumPy, scikit-learn
- **Testing**: Pytest, Hypothesis
- **Documentation**: OpenAPI/Swagger
- **Type Checking**: MyPy with strict typing
- **Data Version Control**: DVC for dataset and model versioning
- **Configuration Management**: Dynaconf for environment-specific settings

### Infrastructure

- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Cloud**: AWS/Azure/GCP/Hetzner (flexible approach)
- **Monitoring & Tracking**:
  - **System Monitoring**: Prometheus, Grafana
  - **ML Experiment Tracking**: MLflow and/or Neptune
  - **Model Registry**: MLflow Models
  - **Alerting**: Prometheus Alertmanager
  - **Distributed Tracing**: OpenTelemetry
  - **ML Model Monitoring**: Evidently for model drift detection and performance
  - **Model Deployment**: Kubeflow for scalable ML pipelines and serving

## Implementation Details

### Code Quality & Testing Strategy

#### Type Safety

- Use strict MyPy type checking across the codebase.
- Implement custom `TypedDict` classes for complex data structures.
- Create type-safe interfaces for external libraries.
- Add runtime type validation for critical functions (e.g., Pydantic for APIs).

#### Testing Approach

- Unit tests for all business logic components (target: >85% coverage for new/critical code).
- Integration tests for API endpoints and data pipelines.
- Performance benchmarks for critical operations.
- Property-based testing for data transformations (`hypothesis`).
- End-to-end tests for complete user workflows (API-driven).
- Focus on testing interfaces and contracts between components.

#### CI/CD Pipeline

- Run tests on multiple Python versions (e.g., 3.11, 3.12).
- Automatically run linting (`ruff`) and type checking (`mypy`).
- Generate and publish coverage reports.
- Build and test Docker images.
- Implement automated deployment to staging/production environments.
- Use caching to speed up build times.

#### Pragmatic Quality Approach

- Address critical issues (syntax errors, security warnings) first.
- Gradually introduce fixes and improvements in focused PRs.
- Use `git commit --no-verify` sparingly and only for genuine work-in-progress commits during initial refactoring/setup phases if pre-commit hooks are temporarily too strict.

### Architecture & Design Patterns

#### Clean Architecture

- Clear separation between domain, application, and infrastructure layers.
- Use dependency injection for flexible component composition (e.g., FastAPI's DI).
- Implement repository pattern for data access abstractions.
- Define clear interfaces (`protocols.py`) between components.

#### SOLID Principles

- **Single Responsibility**: Each module/class has one primary reason to change.
- **Open/Closed**: Extend functionality via interfaces/abstractions without modifying existing stable code.
- **Liskov Substitution**: Subtypes are substitutable for their base types without altering correctness.
- **Interface Segregation**: Prefer smaller, specific interfaces over large, general ones.
- **Dependency Inversion**: Depend on abstractions (interfaces/protocols), not concrete implementations.

#### Error Handling

- Consistent error handling patterns across the codebase (e.g., custom exceptions mapped to HTTP errors in FastAPI).
- Custom exception hierarchy for domain-specific errors.
- Proper logging with contextual information (e.g., request IDs).
- Graceful degradation of functionality where appropriate.

### Data Processing Pipeline

#### Polars Integration

- Use lazy evaluation (`.lazy()`) for efficient processing where applicable.
- Implement custom UDFs carefully, considering performance implications.
- Design for potential streaming processing for very large datasets (future).
- Build caching layer for frequently used calculations (e.g., FastAPI Cache).

#### ML Workflow

- Standardized pipeline for model training and evaluation (potentially using `scikit-learn` Pipelines or custom orchestration).
- Experiment tracking with MLflow/Neptune, logging parameters, metrics, and artifacts.
- Model versioning and reproducibility using MLflow Model Registry and DVC.
- Feature store for reusable transformations (consider Feast or simpler custom implementation).

### Monitoring Strategy (Remains largely the same as original, backend focused)

#### Prometheus Implementation

- **Metrics Collection**:
  - API endpoint latency (FastAPI middleware) and error rates.
  - Resource utilization (CPU, memory, disk, network) via node exporter/cAdvisor.
    // ... rest of metrics ...

#### Grafana Implementation

- **Dashboard Strategy**:

  - Executive overview dashboard for high-level system health
  - Operational dashboards for DevOps and SRE teams
  - Developer dashboards with detailed component metrics
  - ML-specific dashboards for model performance tracking
  - Custom dashboards for specific business domains

- **Key Visualizations**:

  - Multi-stage latency breakdowns for request processing
  - Time-series correlation between system metrics and business KPIs
  - Heatmaps for detection of resource usage patterns
  - Anomaly detection panels with ML-powered forecasting
  - SLO tracking with error budgets and burn rates

- **Advanced Features**:

  - Automated dashboard provisioning via Terraform/GitOps
  - Dashboard-as-code with version control
  - Dynamic variables for environment and service selection
  - Annotation support for deployments and incidents
  - Alert integration with custom notification channels

- **Integration Points**:
  - MLflow metrics visualization in custom panels
  - Business KPI correlation with system metrics
  - User experience metrics from frontend applications
  - Log correlation with Loki integration
  - Custom plugin development for domain-specific visualizations

#### OpenTelemetry Integration

- **Distributed Tracing**:

  - End-to-end request flows
  - Database query performance
  - External API dependencies
  - Model inference latency breakdown
  - User session analysis

- **Logging Integration**:
  - Correlation IDs across services
  - Structured logging for automated analysis
  - Log level management
  - Retention and compliance policies

### Observability Design

- Implement instrumentation as a cross-cutting concern (e.g., FastAPI middleware for tracing/logging).
- Create custom metrics for domain-specific operations using Prometheus client libraries.
- Design health check endpoints for all services (`/health`).
- Implement circuit breakers with metrics for dependencies (e.g., external APIs).
- Configure proper log levels and sampling strategies (especially for tracing).

### Solo Developer Approach (Full Stack Data Scientist - Backend Focus)

As a Full Stack Data Scientist working solo on this **backend** project:

**Core Responsibilities**:

- Backend development (FastAPI, data processing pipelines)
- ML modeling and algorithm implementation
- Infrastructure setup and maintenance (Docker, CI/CD, basic cloud deployment)
- Testing and quality assurance for the backend
- API design and documentation for frontend consumption

**Recommended Focus Order**:

1.  Foundation & Code Quality (Phase 1)
2.  Data pipeline and processing (Phase 2 - highest business value for backend)
3.  Core forecasting capabilities & API exposure (Phase 3)
4.  Infrastructure and deployment automation (Ongoing, solidified in Phase 5)

**Skills Leverage**:

- Leverage FastAPI's features for rapid API development.
- Utilize managed cloud services where possible (e.g., managed databases, object storage) to reduce DevOps burden.
- Prioritize well-documented, battle-tested libraries (Darts, Polars, Pydantic, etc.).
- Implement automation for testing and deployment early via GitHub Actions.

**Outsourcing/Collaboration Considerations**:

- Complex frontend UI development (handled by separate team/repo).
- Advanced infrastructure setup (e.g., complex Kubernetes, service mesh).
- Security auditing.
- Advanced performance optimization.

This focused approach allows for incremental delivery of the backend API while managing the workload effectively.

## Transfer Learning for Time Series

### Adaptive Approach for Unknown Data Types

Unlike computer vision or NLP where pre-trained models on standard datasets are widely applicable, time series transfer learning requires a more nuanced approach when building a general-purpose platform:

1. **Domain-Agnostic Model Features**:

   - Focus on extracting universal time series patterns (seasonality, trend components)
   - Leverage models like N-BEATS that learn basis functions applicable across domains
   - Use decomposition approaches that separate universal components from domain-specific ones

2. **One-Shot Learning Strategy**:

   - Implement "model zoos" with quick-fitting base models (ARIMA, Prophet, simple NN models)
   - Create rapid evaluation framework to quickly identify which model class works best
   - Implement automatic model selection based on limited data characteristics
   - Use fast approximate hyperparameter optimization for quick adaptation

3. **When Pre-Trained Models Actually Help**:

   - For similar domains (e.g., retail sales data, energy consumption)
   - When dealing with low-frequency data with limited history
   - For capturing common patterns like day-of-week, holidays, or annual seasonality
   - As initialization points for faster convergence rather than complete solutions

4. **Practical Implementation**:

   ```python
   # Example: Adaptive model selection based on data characteristics
   from darts.models import NBEATSModel, ExponentialSmoothing, AutoARIMA

   def select_and_train_model(time_series, characteristics):
       """Select appropriate model based on series characteristics"""
       if characteristics['seasonality_strength'] > 0.5:
           # Strong seasonality - try N-BEATS with seasonal configuration
           model = NBEATSModel(
               input_chunk_length=24,
               output_chunk_length=12,
               generic_architecture=False  # Use interpretable mode with trend/seasonality
           )
       elif characteristics['series_length'] < 100:
           # Short series - use simpler model
           model = ExponentialSmoothing()
       else:
           # Default to ARIMA for general cases
           model = AutoARIMA()

       # Train with appropriate settings
       model.fit(time_series)
       return model
   ```

5. **Alternative to Traditional Transfer Learning**:
   - Focus on meta-learning approaches (learning to learn quickly from limited data)
   - Implement adaptive preprocessing based on data characteristics
   - Develop a library of common transformations applicable to specific data types
   - Use automated feature engineering instead of transferred representations

This flexible approach creates more value for a general-purpose platform where uploaded data types are unknown, compared to traditional transfer learning which works best with known domains.

## Polars and ML Tracking Integration

### Why Polars?

- **PySpark Compatibility**: Polars uses a similar lazy-execution model and API as PySpark, making future scaling easier
- **Performance**: Up to 10x faster than Pandas for many operations
- **Memory Efficiency**: Uses Arrow memory format for lower memory usage
- **Scalability**: Better suited for larger datasets
- **Modern API**: Functional API design with method chaining
- **Type Safety**: Better compatibility with strict typing systems
- **Hybrid Mode**: Ability to switch between Polars and PySpark for extreme scale

### Scalability Strategy

- **Abstraction Layer**:

  - Common interface regardless of backend
  - Automatic switching based on data size
  - Configurable thresholds for execution engine selection

- **Large Dataset Handling**:

  - Chunked processing for datasets exceeding memory
  - Incremental processing capabilities
  - Out-of-core computation strategies
  - Cloud storage integration for very large datasets

- **Performance Optimization**:
  - Query optimization for complex operations
  - Parallel execution on multi-core systems
  - Caching strategies for repeated operations
  - Memory-efficient feature transformation

### MLflow & Neptune Benefits

- **Experiment Tracking**: Track all model parameters, metrics, and artifacts
- **Model Registry**: Version and manage models centrally
- **Reproducibility**: Record all data transformations and training steps
- **Collaboration**: Share experiments and results among team members
- **Deployment**: Streamline model deployment to production
- **Compliance**: Maintain audit trail for model development
- **Integration**: Connect with Evidently for continuous model monitoring
- **Transfer Learning**: Track performance improvements from pre-trained models

### DVC Implementation

- **Dataset Versioning**:

  - Track changes to datasets over time
  - Store large files efficiently
  - Connect data versions to model versions
  - Enable reproducible experiments

- **Pipeline Management**:

  - Define and version data processing pipelines
  - Track dependencies between stages
  - Automate pipeline execution and validation
  - Integrate with CI/CD for automated testing

- **Collaboration Features**:
  - Share datasets and models across team members
  - Review changes to data and models
  - Integrate with cloud storage for scalable sharing
  - Compare results across experiments

## Questions to Consider

// ... existing Questions to Consider content ...

## Record of Achievement (Completed Tasks)

### Initial Prototype Setup & Configuration:

- Basic Streamlit interface implemented
- Initial data upload functionality
- Simple time series visualization
- Development environment with UV for dependency management
- Type-safe session state management for Streamlit (to be reviewed for backend focus)
- MyPy configured for static type checking with pre-commit hooks
- Package configuration fixed in `pyproject.toml`
