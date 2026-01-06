/**
 * HomePage Component
 * Landing page with file upload and recent datasets
 */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { FileUpload } from '@/components/FileUpload';
import { datasetApi, Dataset } from '@/services/api';

export function HomePage() {
  const navigate = useNavigate();

  // Get recent datasets
  const { data: datasets, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: datasetApi.getAll,
  });

  const handleUploadSuccess = (dataset: Dataset) => {
    setTimeout(() => {
      navigate(`/analysis/${dataset.dataset_id}`);
    }, 1500);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">D</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">DataPrep AI</h1>
                <p className="text-xs text-slate-500">Intelligent Data Preprocessing</p>
              </div>
            </div>
            <nav className="flex items-center gap-4">
              <a href="/docs" className="text-sm text-slate-600 hover:text-slate-900 transition-colors">
                Documentation
              </a>
              <a href="/help" className="text-sm text-slate-600 hover:text-slate-900 transition-colors">
                Help
              </a>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12">
        {/* Hero Section */}
        <section className="text-center mb-16">
          <h2 className="text-4xl font-bold text-slate-900 mb-4">
            Automate Your Data Preprocessing
          </h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto mb-8">
            Upload your CSV dataset and let our AI-powered platform analyze, clean,
            and transform your data. Reduce preprocessing time by up to 70%.
          </p>

          {/* Feature badges */}
          <div className="flex flex-wrap justify-center gap-3 mb-12">
            {[
              '🔍 Automatic Profiling',
              '✨ Quality Assessment',
              '🔄 Smart Transformations',
              '📊 Interactive Visualizations',
              '📝 Code Generation',
            ].map((feature) => (
              <span
                key={feature}
                className="px-4 py-2 bg-white rounded-full text-sm text-slate-700 shadow-sm border border-slate-200"
              >
                {feature}
              </span>
            ))}
          </div>
        </section>

        {/* Upload Section */}
        <section className="max-w-2xl mx-auto mb-16">
          <div className="bg-white rounded-2xl shadow-lg p-8 border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-900 mb-6 text-center">
              Get Started
            </h3>
            <FileUpload onUploadSuccess={handleUploadSuccess} />
          </div>
        </section>

        {/* Recent Datasets Section */}
        {datasets && datasets.length > 0 && (
          <section className="max-w-4xl mx-auto">
            <h3 className="text-xl font-semibold text-slate-900 mb-6">
              Recent Datasets
            </h3>
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-slate-200">
              <table className="w-full">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">
                      Rows × Cols
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">
                      Uploaded
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-slate-500 uppercase">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {datasets.slice(0, 5).map((dataset) => (
                    <tr key={dataset.dataset_id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center">
                            <span className="text-indigo-600 text-sm">📊</span>
                          </div>
                          <span className="font-medium text-slate-900">{dataset.filename}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        {formatFileSize(dataset.file_size_bytes)}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        {dataset.num_rows?.toLocaleString()} × {dataset.num_columns}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-500">
                        {formatDate(dataset.upload_date)}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => navigate(`/analysis/${dataset.dataset_id}`)}
                          className="px-4 py-2 text-sm font-medium text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 rounded-lg transition-colors"
                        >
                          Analyze →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Loading state */}
        {isLoading && (
          <div className="text-center text-slate-500">
            Loading recent datasets...
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto">
        <div className="container mx-auto px-6 py-6 text-center text-sm text-slate-500">
          DataPrep AI Platform • Powered by Python, FastAPI, and React
        </div>
      </footer>
    </div>
  );
}

export default HomePage;