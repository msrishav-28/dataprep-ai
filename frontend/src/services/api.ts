/**
 * API Client for DataPrep AI Backend
 * Centralized API communication using Axios
 */
import axios, { AxiosInstance } from 'axios';

// API Configuration
const API_BASE_URL = '/api/v1';

// Create Axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred';
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);

// Types
export interface Dataset {
  dataset_id: string;
  filename: string;
  file_path: string;
  file_size_bytes: number;
  num_rows: number;
  num_columns: number;
  upload_date: string;
  status: string;
}

export interface QualityAssessment {
  assessment_timestamp: string;
  dataset_summary: {
    num_rows: number;
    num_columns: number;
    memory_usage_mb: number;
  };
  quality_scores: {
    overall: number;
    completeness: number;
    uniqueness: number;
    consistency: number;
    validity: number;
  };
  issues: Array<{
    issue_id: string;
    issue_type: string;
    severity: string;
    column: string | null;
    description: string;
    affected_rows: number;
    affected_percentage: number;
    recommendation: string;
  }>;
  issue_summary: {
    total_issues: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
  };
  column_quality?: Record<string, {
    quality_score: number;
    completeness: number;
    missing_count: number;
  }>;
}

export interface TransformationRequest {
  transformation_type: string;
  columns: string[];
  constant_value?: any;
  threshold?: number;
  iqr_multiplier?: number;
  target_dtype?: string;
  new_name?: string;
}

// Dataset API
export const datasetApi = {
  // Upload a new dataset
  upload: async (file: File, onProgress?: (progress: number) => void): Promise<Dataset> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<Dataset>('/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
    return response.data;
  },

  // Get dataset by ID
  getById: async (datasetId: string): Promise<Dataset> => {
    const response = await apiClient.get<Dataset>(`/datasets/${datasetId}`);
    return response.data;
  },

  // Get all datasets
  getAll: async (): Promise<Dataset[]> => {
    const response = await apiClient.get<Dataset[]>('/datasets/');
    return response.data;
  },

  // Get dataset preview (sample rows)
  getPreview: async (datasetId: string, rows: number = 100): Promise<any> => {
    const response = await apiClient.get(`/datasets/${datasetId}/preview?num_rows=${rows}`);
    return response.data;
  },

  // Delete dataset
  delete: async (datasetId: string): Promise<void> => {
    await apiClient.delete(`/datasets/${datasetId}`);
  },
};

// Analysis API
export const analysisApi = {
  // Get dataset profile
  getProfile: async (datasetId: string): Promise<any> => {
    const response = await apiClient.get(`/analyze/profile/${datasetId}`);
    return response.data;
  },

  // Get quality assessment
  getQualityAssessment: async (datasetId: string): Promise<QualityAssessment> => {
    const response = await apiClient.get<QualityAssessment>(`/analyze/quality/${datasetId}`);
    return response.data;
  },

  // Get visualizations
  getVisualizations: async (datasetId: string): Promise<any> => {
    const response = await apiClient.get(`/analyze/visualizations/${datasetId}`);
    return response.data;
  },

  // Start background analysis
  startAnalysis: async (datasetId: string): Promise<{ task_id: string }> => {
    const response = await apiClient.post(`/analyze/start/${datasetId}`);
    return response.data;
  },

  // Get analysis task status
  getTaskStatus: async (taskId: string): Promise<any> => {
    const response = await apiClient.get(`/analyze/task/${taskId}`);
    return response.data;
  },
};

// Transformation API
export const transformApi = {
  // Preview transformation
  preview: async (datasetId: string, request: TransformationRequest): Promise<any> => {
    const response = await apiClient.post(`/transform/preview/${datasetId}`, request);
    return response.data;
  },

  // Apply transformation
  apply: async (datasetId: string, request: TransformationRequest): Promise<any> => {
    const response = await apiClient.post(`/transform/apply/${datasetId}`, request);
    return response.data;
  },

  // Get transformation history
  getHistory: async (datasetId: string): Promise<any> => {
    const response = await apiClient.get(`/transform/history/${datasetId}`);
    return response.data;
  },

  // Undo last transformation
  undo: async (datasetId: string): Promise<any> => {
    const response = await apiClient.post(`/transform/undo/${datasetId}`);
    return response.data;
  },

  // Get available transformation types
  getTypes: async (): Promise<any> => {
    const response = await apiClient.get('/transform/types');
    return response.data;
  },
};

// Export API
export const exportApi = {
  // Get Python code
  getCode: async (datasetId: string, style: 'pandas' | 'sklearn' = 'pandas'): Promise<string> => {
    const response = await apiClient.get(`/export/code/${datasetId}?style=${style}`, {
      responseType: 'text',
    });
    return response.data;
  },

  // Get Jupyter notebook
  getNotebook: async (datasetId: string): Promise<any> => {
    const response = await apiClient.get(`/export/notebook/${datasetId}`);
    return response.data;
  },

  // Download cleaned data
  downloadData: (datasetId: string): string => {
    return `${API_BASE_URL}/export/data/${datasetId}`;
  },

  // Download HTML report
  downloadReport: (datasetId: string): string => {
    return `${API_BASE_URL}/export/report/${datasetId}`;
  },

  // Get export options
  getOptions: async (datasetId: string): Promise<any> => {
    const response = await apiClient.get(`/export/summary/${datasetId}`);
    return response.data;
  },
};

export default apiClient;
