# Frontend Documentation

## Tech Stack
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![React Query](https://img.shields.io/badge/-React%20Query-FF4154?style=for-the-badge&logo=react%20query&logoColor=white)
![React Router](https://img.shields.io/badge/React_Router-CA4245?style=for-the-badge&logo=react-router&logoColor=white)

---

## Components

### FileUpload (`components/FileUpload.tsx`)
Drag-and-drop CSV upload with:
- Progress indicator
- File type validation
- Error handling
- Success callback

### QualityCards (`components/QualityCards.tsx`)
Quality display components:
- `QualityScoreCard` - Score with progress bar
- `IssueCard` - Issue with severity badge
- `StatCard` - Generic stat display

### TransformationPanel (`components/TransformationPanel.tsx`)
Transformation interface with:
- Transformation type selector
- Before/after comparison
- Undo button
- History display

### RecommendationCard (`components/RecommendationCard.tsx`)
Educational recommendations with:
- Priority badges
- Expandable "Why?" explanations
- Code snippets
- Alternative approaches

---

## Pages

### HomePage (`pages/HomePage.tsx`)
Landing page with:
- Hero section
- Feature badges
- File upload component
- Recent datasets list

### AnalysisPage (`pages/AnalysisPage.tsx`)
Dashboard with tabs:
- **Overview** - Quality scores, issue summary, data preview
- **Quality** - All detected issues
- **Columns** - Per-column statistics
- **Transform** - Transformation tools
- **Export** - Download options

---

## API Client (`services/api.ts`)
Typed API client with:
- Axios instance with interceptors
- Dataset, Analysis, Transform, Export API modules
- TypeScript interfaces for all responses

---

## Configuration

### Path Aliases
`@/*` maps to `src/*` (configured in `tsconfig.json` and `vite.config.ts`)

### Proxy
Development server proxies `/api` to `http://localhost:8000`

---

## Scripts

```bash
npm run dev      # Start dev server (port 3000)
npm run build    # Production build
npm run preview  # Preview production build
npm run lint     # Run ESLint
```
