# DataPrep AI

**Intelligent Data Preprocessing & EDA Platform**

Transform data preparation from hours to minutes with automated analysis, transformation, and code generation.

---

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Access: http://localhost:3000

---

## Features

| Feature | Description |
|---------|-------------|
| **Auto Profiling** | Statistical analysis, distribution detection, type inference |
| **Quality Assessment** | Missing values, duplicates, outliers, type inconsistencies |
| **Smart Recommendations** | Educational "Why?" explanations for each suggestion |
| **Transformations** | Imputation, encoding, scaling, outlier treatment with undo |
| **Code Generation** | Export Python scripts and Jupyter notebooks |
| **HTML Reports** | Self-contained analysis reports |

---

## Architecture

```
dataprep-ai/
├── backend/                 # FastAPI Python backend
│   ├── app/
│   │   ├── api/v1/         # REST API endpoints
│   │   ├── core/           # Config, database, security
│   │   ├── models/         # SQLAlchemy models
│   │   ├── schemas/        # Pydantic validation
│   │   └── services/       # Business logic
│   └── tests/              # Pytest tests
├── frontend/               # React TypeScript frontend
│   └── src/
│       ├── components/     # Reusable UI components
│       ├── pages/          # Page components
│       └── services/       # API client
└── docker-compose.yml      # Container orchestration
```

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Get JWT token |
| GET | `/api/v1/auth/me` | Current user |

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/datasets/upload` | Upload CSV |
| GET | `/api/v1/datasets/{id}` | Get metadata |
| GET | `/api/v1/datasets/{id}/preview` | Sample rows |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analyze/quality/{id}` | Quality assessment |
| GET | `/api/v1/analyze/visualizations/{id}` | Chart configs |

### Transformations
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/transform/preview/{id}` | Preview change |
| POST | `/api/v1/transform/apply/{id}` | Apply change |
| POST | `/api/v1/transform/undo/{id}` | Undo last |

### Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/export/code/{id}` | Python script |
| GET | `/api/v1/export/notebook/{id}` | Jupyter notebook |
| GET | `/api/v1/export/report/{id}` | HTML report |

---

## Tech Stack

### Backend
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

### Frontend
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![React Query](https://img.shields.io/badge/-React%20Query-FF4154?style=for-the-badge&logo=react%20query&logoColor=white)

### Infrastructure
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![Celery](https://img.shields.io/badge/celery-%2337814A.svg?style=for-the-badge&logo=celery&logoColor=white)

---

## Configuration

Create `.env` in `/backend`:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/dataprep_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
```

---

## License

MIT License
