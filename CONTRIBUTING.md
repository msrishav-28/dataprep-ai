# Contributing Guide

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+ (or SQLite for development)
- Redis (optional, for background tasks)

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/   # API routes
│   ├── core/               # Config, DB, security
│   ├── models/             # SQLAlchemy models
│   ├── schemas/            # Pydantic schemas
│   └── services/           # Business logic
└── tests/                  # Pytest tests

frontend/
└── src/
    ├── components/         # Reusable components
    ├── pages/              # Route pages
    └── services/           # API client
```

---

## Coding Standards

### Python
- Black for formatting
- isort for imports
- Type hints required
- Docstrings for public functions

### TypeScript
- ESLint + Prettier
- Strict mode enabled
- Interfaces for API types

---

## Adding a New Transformation

1. Add type to `TransformationType` enum in `transformation_service.py`
2. Implement `_apply_<name>` method
3. Add to `_get_handler` dispatcher
4. Update `/transform/types` response
5. Add property test

---

## Running Tests

```bash
# Backend
cd backend
pytest tests/ -v

# With coverage
pytest --cov=app tests/
```
