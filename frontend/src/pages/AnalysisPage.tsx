/**
 * AnalysisPage Component
 * Main dashboard for dataset analysis results
 */
import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { datasetApi, analysisApi, exportApi } from '@/services/api';
import { QualityScoreCard, IssueCard, StatCard } from '@/components/QualityCards';

type TabId = 'overview' | 'quality' | 'columns' | 'transform' | 'export';

export function AnalysisPage() {
    const { datasetId } = useParams<{ datasetId: string }>();
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState<TabId>('overview');

    // Fetch dataset info
    const { data: dataset, isLoading: datasetLoading } = useQuery({
        queryKey: ['dataset', datasetId],
        queryFn: () => datasetApi.getById(datasetId!),
        enabled: !!datasetId,
    });

    // Fetch quality assessment
    const { data: quality, isLoading: qualityLoading } = useQuery({
        queryKey: ['quality', datasetId],
        queryFn: () => analysisApi.getQualityAssessment(datasetId!),
        enabled: !!datasetId,
    });

    // Fetch data preview
    const { data: preview } = useQuery({
        queryKey: ['preview', datasetId],
        queryFn: () => datasetApi.getPreview(datasetId!, 50),
        enabled: !!datasetId,
    });

    const isLoading = datasetLoading || qualityLoading;

    const tabs: { id: TabId; label: string; icon: string }[] = [
        { id: 'overview', label: 'Overview', icon: '📊' },
        { id: 'quality', label: 'Quality', icon: '✅' },
        { id: 'columns', label: 'Columns', icon: '📋' },
        { id: 'transform', label: 'Transform', icon: '🔄' },
        { id: 'export', label: 'Export', icon: '📤' },
    ];

    if (isLoading) {
        return (
            <div className="min-h-screen bg-slate-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-slate-600">Analyzing dataset...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-50">
            {/* Header */}
            <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
                <div className="container mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <button
                                onClick={() => navigate('/')}
                                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                            >
                                ← Back
                            </button>
                            <div>
                                <h1 className="text-xl font-bold text-slate-900">
                                    {dataset?.filename || 'Dataset Analysis'}
                                </h1>
                                <p className="text-sm text-slate-500">
                                    {dataset?.num_rows?.toLocaleString()} rows × {dataset?.num_columns} columns
                                </p>
                            </div>
                        </div>

                        {/* Quality badge */}
                        {quality && (
                            <div className={`
                px-4 py-2 rounded-full font-medium text-sm
                ${quality.quality_scores.overall >= 80 ? 'bg-green-100 text-green-700' :
                                    quality.quality_scores.overall >= 60 ? 'bg-yellow-100 text-yellow-700' :
                                        'bg-red-100 text-red-700'}
              `}>
                                Quality: {quality.quality_scores.overall.toFixed(0)}%
                            </div>
                        )}
                    </div>

                    {/* Tabs */}
                    <nav className="flex gap-1 mt-4 -mb-px">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`
                  px-4 py-2 text-sm font-medium rounded-t-lg transition-colors
                  ${activeTab === tab.id
                                        ? 'bg-indigo-50 text-indigo-600 border-b-2 border-indigo-500'
                                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}
                `}
                            >
                                {tab.icon} {tab.label}
                            </button>
                        ))}
                    </nav>
                </div>
            </header>

            <main className="container mx-auto px-6 py-8">
                {/* Overview Tab */}
                {activeTab === 'overview' && quality && (
                    <div className="space-y-8">
                        {/* Quality Scores Grid */}
                        <section>
                            <h2 className="text-lg font-semibold text-slate-900 mb-4">Quality Scores</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                                <QualityScoreCard
                                    title="Overall"
                                    score={quality.quality_scores.overall}
                                    description="Combined quality metric"
                                />
                                <QualityScoreCard
                                    title="Completeness"
                                    score={quality.quality_scores.completeness}
                                    description="Non-missing values"
                                />
                                <QualityScoreCard
                                    title="Uniqueness"
                                    score={quality.quality_scores.uniqueness}
                                    description="Non-duplicate rows"
                                />
                                <QualityScoreCard
                                    title="Consistency"
                                    score={quality.quality_scores.consistency}
                                    description="Data type consistency"
                                />
                                <QualityScoreCard
                                    title="Validity"
                                    score={quality.quality_scores.validity}
                                    description="Values within bounds"
                                />
                            </div>
                        </section>

                        {/* Issue Summary */}
                        <section>
                            <h2 className="text-lg font-semibold text-slate-900 mb-4">
                                Issues Detected ({quality.issue_summary.total_issues})
                            </h2>
                            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                                {[
                                    { label: 'Critical', count: quality.issue_summary.critical, color: 'red' },
                                    { label: 'High', count: quality.issue_summary.high, color: 'orange' },
                                    { label: 'Medium', count: quality.issue_summary.medium, color: 'yellow' },
                                    { label: 'Low', count: quality.issue_summary.low, color: 'green' },
                                    { label: 'Info', count: quality.issue_summary.info, color: 'blue' },
                                ].map(({ label, count, color }) => (
                                    <div key={label} className={`p-4 rounded-lg bg-${color}-50 border border-${color}-200`}>
                                        <p className={`text-2xl font-bold text-${color}-600`}>{count}</p>
                                        <p className={`text-sm text-${color}-700`}>{label}</p>
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* Data Preview */}
                        {preview?.data && (
                            <section>
                                <h2 className="text-lg font-semibold text-slate-900 mb-4">Data Preview</h2>
                                <div className="bg-white rounded-xl border border-slate-200 overflow-auto">
                                    <table className="w-full text-sm">
                                        <thead className="bg-slate-50 border-b border-slate-200">
                                            <tr>
                                                {preview.columns?.map((col: string) => (
                                                    <th key={col} className="px-4 py-3 text-left font-medium text-slate-600">
                                                        {col}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {preview.data?.slice(0, 10).map((row: any, idx: number) => (
                                                <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                                                    {preview.columns?.map((col: string) => (
                                                        <td key={col} className="px-4 py-2 text-slate-700 max-w-xs truncate">
                                                            {row[col]?.toString() || '—'}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        )}
                    </div>
                )}

                {/* Quality Tab */}
                {activeTab === 'quality' && quality && (
                    <div className="space-y-4">
                        <h2 className="text-lg font-semibold text-slate-900 mb-4">All Quality Issues</h2>
                        {quality.issues.length === 0 ? (
                            <div className="text-center py-12 bg-white rounded-xl border border-slate-200">
                                <span className="text-4xl mb-4 block">✅</span>
                                <p className="text-slate-600">No quality issues detected!</p>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {quality.issues.map((issue) => (
                                    <IssueCard key={issue.issue_id} issue={issue} />
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Columns Tab */}
                {activeTab === 'columns' && quality?.column_quality && (
                    <div className="space-y-4">
                        <h2 className="text-lg font-semibold text-slate-900 mb-4">Column Analysis</h2>
                        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                            <table className="w-full">
                                <thead className="bg-slate-50 border-b border-slate-200">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Column</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Quality</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Completeness</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Missing</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {Object.entries(quality.column_quality).map(([col, info]: [string, any]) => (
                                        <tr key={col} className="hover:bg-slate-50">
                                            <td className="px-4 py-3 font-medium text-slate-900">{col}</td>
                                            <td className="px-4 py-3">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-16 h-2 bg-slate-200 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full ${info.quality_score >= 80 ? 'bg-green-500' : info.quality_score >= 60 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                                            style={{ width: `${info.quality_score}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm text-slate-600">{info.quality_score?.toFixed(0)}%</span>
                                                </div>
                                            </td>
                                            <td className="px-4 py-3 text-sm text-slate-600">{info.completeness?.toFixed(1)}%</td>
                                            <td className="px-4 py-3 text-sm text-slate-600">{info.missing_count}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Export Tab */}
                {activeTab === 'export' && datasetId && (
                    <div className="space-y-6">
                        <h2 className="text-lg font-semibold text-slate-900 mb-4">Export Options</h2>
                        <div className="grid md:grid-cols-2 gap-4">
                            {[
                                { title: 'Python Code', desc: 'Download preprocessing script', icon: '🐍', url: exportApi.downloadData(datasetId).replace('/data/', '/code/') },
                                { title: 'Jupyter Notebook', desc: 'Interactive notebook format', icon: '📓', url: `/api/v1/export/notebook/${datasetId}` },
                                { title: 'Cleaned Data', desc: 'Download processed CSV', icon: '📊', url: exportApi.downloadData(datasetId) },
                                { title: 'HTML Report', desc: 'Comprehensive analysis report', icon: '📄', url: exportApi.downloadReport(datasetId) },
                            ].map((option) => (
                                <a
                                    key={option.title}
                                    href={option.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="block p-6 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-md transition-all"
                                >
                                    <span className="text-3xl mb-3 block">{option.icon}</span>
                                    <h3 className="font-semibold text-slate-900">{option.title}</h3>
                                    <p className="text-sm text-slate-500">{option.desc}</p>
                                </a>
                            ))}
                        </div>
                    </div>
                )}

                {/* Transform Tab Placeholder */}
                {activeTab === 'transform' && (
                    <div className="text-center py-12 bg-white rounded-xl border border-slate-200">
                        <span className="text-4xl mb-4 block">🔄</span>
                        <h3 className="text-lg font-semibold text-slate-900 mb-2">Transformation Tools</h3>
                        <p className="text-slate-600">
                            Apply transformations like imputation, outlier treatment, encoding, and scaling.
                        </p>
                    </div>
                )}
            </main>
        </div>
    );
}

export default AnalysisPage;
