# Backend Documentation

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

## Services

### Quality Service (`quality_service.py`)
Detects data quality issues:
- Missing values with severity levels
- Duplicate rows (exact and near-duplicates)
- Outliers using Z-score and IQR methods
- Type inconsistencies and mixed types
- High cardinality and constant columns

### Transformation Service (`transformation_service.py`)
Data transformation engine with:
- **Imputation:** Mean, median, mode, constant, forward/backward fill
- **Outliers:** Remove or cap using Z-score/IQR
- **Encoding:** One-hot, label encoding
- **Scaling:** StandardScaler, MinMaxScaler
- **Undo support:** Full history tracking

### Visualization Service (`visualization_service.py`)
Plotly chart generation:
- Histograms (numerical distributions)
- Box plots (outlier visualization)
- Correlation heatmaps
- Missing value patterns
- Categorical bar charts

### Code Generator (`code_generator.py`)
Exports preprocessing pipelines:
- Pandas-style Python scripts
- Scikit-learn Pipeline objects
- Jupyter notebook format

### Recommendation Engine (`recommendation_engine.py`)
Intelligent suggestions with:
- Priority ranking (critical/high/medium/low)
- Educational "Why?" explanations
- Code snippets for each fix
- Alternative approaches

### Report Generator (`report_generator.py`)
Self-contained HTML reports:
- Dataset overview
- Quality scores
- Column statistics
- Embedded styling

---

## API Endpoints

### Auth (`auth.py`)
- `POST /register` - Create user
- `POST /login` - Get JWT token
- `GET /me` - Current user profile

### Datasets (`datasets.py`)
- `POST /upload` - Upload CSV file
- `GET /{id}` - Dataset metadata
- `GET /{id}/preview` - Sample rows
- `DELETE /{id}` - Remove dataset

### Analysis (`analysis.py`)
- `GET /quality/{id}` - Quality assessment
- `GET /visualizations/{id}` - Chart data

### Transform (`transform.py`)
- `POST /preview/{id}` - Preview transformation
- `POST /apply/{id}` - Apply transformation
- `GET /history/{id}` - Transformation history
- `POST /undo/{id}` - Undo last change

### Export (`export.py`)
- `GET /code/{id}` - Python script
- `GET /notebook/{id}` - Jupyter notebook
- `GET /data/{id}` - Cleaned CSV
- `GET /report/{id}` - HTML report

### Pipelines (`pipelines.py`)
- `POST /` - Save pipeline
- `GET /` - List user pipelines
- `POST /{id}/apply/{dataset_id}` - Apply to dataset

---

## Models

### User
```python
user_id: UUID
email: str
password_hash: str
full_name: str
subscription_tier: str  # free, pro, team
```

### Dataset
```python
dataset_id: UUID
user_id: UUID (FK)
filename: str
file_path: str
num_rows: int
num_columns: int
status: str  # uploaded, processing, ready
```

### Pipeline
```python
pipeline_id: UUID
user_id: UUID (FK)
pipeline_name: str
transformations_json: JSONB
is_public: bool
```
