# API Documentation

Complete API reference for DataPrep AI Platform.

Base URL: `http://localhost:8000/api/v1`

---

## Authentication

All protected endpoints require Bearer token:
```
Authorization: Bearer <access_token>
```

### Register
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123",
  "full_name": "John Doe"
}
```

### Login
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=SecurePass123
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

## Datasets

### Upload CSV
```http
POST /datasets/upload
Content-Type: multipart/form-data

file: <csv_file>
```

### Get Dataset
```http
GET /datasets/{dataset_id}
```

### Preview Data
```http
GET /datasets/{dataset_id}/preview?num_rows=100
```

---

## Analysis

### Quality Assessment
```http
GET /analyze/quality/{dataset_id}
```

Response:
```json
{
  "quality_scores": {
    "overall": 85.5,
    "completeness": 92.0,
    "uniqueness": 98.0,
    "consistency": 78.0,
    "validity": 89.0
  },
  "issues": [...],
  "recommendations": [...]
}
```

### Visualizations
```http
GET /analyze/visualizations/{dataset_id}
```

---

## Transformations

### Preview
```http
POST /transform/preview/{dataset_id}
Content-Type: application/json

{
  "transformation_type": "impute_mean",
  "columns": ["age", "salary"]
}
```

### Apply
```http
POST /transform/apply/{dataset_id}
```

### Undo
```http
POST /transform/undo/{dataset_id}
```

### Available Types
```http
GET /transform/types
```

---

## Export

### Python Code
```http
GET /export/code/{dataset_id}?style=pandas
```

### Jupyter Notebook
```http
GET /export/notebook/{dataset_id}
```

### Cleaned Data
```http
GET /export/data/{dataset_id}
```

### HTML Report
```http
GET /export/report/{dataset_id}
```

---

## Pipelines

### Save Pipeline
```http
POST /pipelines
Content-Type: application/json

{
  "pipeline_name": "My Preprocessing",
  "transformations": [...],
  "is_public": false
}
```

### Apply to Dataset
```http
POST /pipelines/{pipeline_id}/apply/{dataset_id}
```
