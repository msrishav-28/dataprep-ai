/**
 * TransformationPanel Component
 * Complete transformation interface with undo/redo and before/after comparison
 */
import React, { useState, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { transformApi } from '@/services/api';

interface TransformationPanelProps {
    datasetId: string;
    onTransformComplete?: () => void;
}

interface TransformationStep {
    id: string;
    transformation_type: string;
    columns: string[];
    parameters: Record<string, any>;
    timestamp: string;
    rows_affected: number;
}

export const TransformationPanel: React.FC<TransformationPanelProps> = ({
    datasetId,
    onTransformComplete,
}) => {
    const queryClient = useQueryClient();
    const [selectedTransform, setSelectedTransform] = useState<string>('');
    const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
    const [previewData, setPreviewData] = useState<any>(null);
    const [showBeforeAfter, setShowBeforeAfter] = useState(false);

    // Fetch transformation history
    const { data: history, refetch: refetchHistory } = useQuery({
        queryKey: ['transform-history', datasetId],
        queryFn: () => transformApi.getHistory(datasetId),
    });

    // Fetch available transformation types
    const { data: transformTypes } = useQuery({
        queryKey: ['transform-types'],
        queryFn: () => transformApi.getTypes(),
    });

    // Preview mutation
    const previewMutation = useMutation({
        mutationFn: (request: any) => transformApi.preview(datasetId, request),
        onSuccess: (data) => {
            setPreviewData(data);
            setShowBeforeAfter(true);
        },
    });

    // Apply mutation
    const applyMutation = useMutation({
        mutationFn: (request: any) => transformApi.apply(datasetId, request),
        onSuccess: () => {
            refetchHistory();
            setPreviewData(null);
            setShowBeforeAfter(false);
            queryClient.invalidateQueries({ queryKey: ['quality', datasetId] });
            queryClient.invalidateQueries({ queryKey: ['preview', datasetId] });
            onTransformComplete?.();
        },
    });

    // Undo mutation
    const undoMutation = useMutation({
        mutationFn: () => transformApi.undo(datasetId),
        onSuccess: () => {
            refetchHistory();
            queryClient.invalidateQueries({ queryKey: ['quality', datasetId] });
            queryClient.invalidateQueries({ queryKey: ['preview', datasetId] });
        },
    });

    const handlePreview = () => {
        if (!selectedTransform) return;
        previewMutation.mutate({
            transformation_type: selectedTransform,
            columns: selectedColumns,
        });
    };

    const handleApply = () => {
        if (!selectedTransform) return;
        applyMutation.mutate({
            transformation_type: selectedTransform,
            columns: selectedColumns,
        });
    };

    const handleUndo = () => {
        undoMutation.mutate();
    };

    const transformationCategories = useMemo(() => {
        if (!transformTypes?.types) return {};
        return transformTypes.types.reduce((acc: any, t: any) => {
            const category = t.category || 'other';
            if (!acc[category]) acc[category] = [];
            acc[category].push(t);
            return acc;
        }, {});
    }, [transformTypes]);

    return (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-slate-900">Transformation Studio</h2>
                <div className="flex items-center gap-2">
                    <button
                        onClick={handleUndo}
                        disabled={!history?.history?.length || undoMutation.isPending}
                        className="px-3 py-1.5 text-sm font-medium text-slate-600 hover:text-slate-900 
                       bg-white border border-slate-200 rounded-lg hover:bg-slate-50
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        ↩ Undo
                    </button>
                    <span className="text-xs text-slate-500">
                        {history?.history?.length || 0} steps
                    </span>
                </div>
            </div>

            <div className="p-6 grid md:grid-cols-2 gap-6">
                {/* Left: Transformation Selection */}
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                            Select Transformation
                        </label>
                        <select
                            value={selectedTransform}
                            onChange={(e) => setSelectedTransform(e.target.value)}
                            className="w-full px-3 py-2 border border-slate-300 rounded-lg 
                         focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                        >
                            <option value="">Choose a transformation...</option>
                            {Object.entries(transformationCategories).map(([category, transforms]: [string, any]) => (
                                <optgroup key={category} label={category.replace('_', ' ').toUpperCase()}>
                                    {transforms.map((t: any) => (
                                        <option key={t.type} value={t.type}>
                                            {t.name}
                                        </option>
                                    ))}
                                </optgroup>
                            ))}
                        </select>
                    </div>

                    {/* Column Selection */}
                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                            Target Columns
                        </label>
                        <div className="max-h-40 overflow-y-auto border border-slate-200 rounded-lg p-2">
                            {/* Column checkboxes would be populated from dataset metadata */}
                            <p className="text-sm text-slate-500 italic">
                                Select columns from dataset...
                            </p>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3">
                        <button
                            onClick={handlePreview}
                            disabled={!selectedTransform || previewMutation.isPending}
                            className="flex-1 px-4 py-2 text-sm font-medium text-indigo-600 
                         bg-indigo-50 hover:bg-indigo-100 rounded-lg 
                         disabled:opacity-50 transition-colors"
                        >
                            {previewMutation.isPending ? 'Loading...' : '👁️ Preview'}
                        </button>
                        <button
                            onClick={handleApply}
                            disabled={!selectedTransform || applyMutation.isPending}
                            className="flex-1 px-4 py-2 text-sm font-medium text-white 
                         bg-indigo-600 hover:bg-indigo-700 rounded-lg 
                         disabled:opacity-50 transition-colors"
                        >
                            {applyMutation.isPending ? 'Applying...' : '✓ Apply'}
                        </button>
                    </div>
                </div>

                {/* Right: Before/After Comparison */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium text-slate-700">
                            Preview Comparison
                        </label>
                        {previewData && (
                            <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full">
                                {previewData.rows_affected} rows affected
                            </span>
                        )}
                    </div>

                    {showBeforeAfter && previewData ? (
                        <div className="grid grid-cols-2 gap-4">
                            {/* Before */}
                            <div className="border border-slate-200 rounded-lg overflow-hidden">
                                <div className="px-3 py-2 bg-red-50 border-b border-red-200">
                                    <span className="text-xs font-medium text-red-700">BEFORE</span>
                                </div>
                                <div className="p-3 text-xs font-mono bg-slate-50 max-h-48 overflow-auto">
                                    {previewData.before_stats && (
                                        <div className="space-y-1">
                                            <p>Mean: {previewData.before_stats.mean?.toFixed(2)}</p>
                                            <p>Null: {previewData.before_stats.null_count}</p>
                                            <p>Min: {previewData.before_stats.min?.toFixed(2)}</p>
                                            <p>Max: {previewData.before_stats.max?.toFixed(2)}</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* After */}
                            <div className="border border-slate-200 rounded-lg overflow-hidden">
                                <div className="px-3 py-2 bg-green-50 border-b border-green-200">
                                    <span className="text-xs font-medium text-green-700">AFTER</span>
                                </div>
                                <div className="p-3 text-xs font-mono bg-slate-50 max-h-48 overflow-auto">
                                    {previewData.after_stats && (
                                        <div className="space-y-1">
                                            <p>Mean: {previewData.after_stats.mean?.toFixed(2)}</p>
                                            <p>Null: {previewData.after_stats.null_count}</p>
                                            <p>Min: {previewData.after_stats.min?.toFixed(2)}</p>
                                            <p>Max: {previewData.after_stats.max?.toFixed(2)}</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="h-48 flex items-center justify-center bg-slate-50 rounded-lg border border-dashed border-slate-300">
                            <p className="text-sm text-slate-400">
                                Select a transformation and click Preview to see changes
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Transformation History */}
            {history?.history && history.history.length > 0 && (
                <div className="px-6 py-4 border-t border-slate-200 bg-slate-50">
                    <h3 className="text-sm font-medium text-slate-700 mb-3">Applied Transformations</h3>
                    <div className="flex flex-wrap gap-2">
                        {history.history.map((step: TransformationStep, index: number) => (
                            <div
                                key={step.id || index}
                                className="px-3 py-1.5 bg-white border border-slate-200 rounded-full 
                           text-xs text-slate-600 flex items-center gap-2"
                            >
                                <span className="w-5 h-5 bg-indigo-100 text-indigo-600 rounded-full 
                               flex items-center justify-center font-medium">
                                    {index + 1}
                                </span>
                                <span>{step.transformation_type.replace(/_/g, ' ')}</span>
                                {step.rows_affected > 0 && (
                                    <span className="text-slate-400">({step.rows_affected} rows)</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default TransformationPanel;
