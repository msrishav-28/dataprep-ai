/**
 * RecommendationCard Component
 * Displays intelligent recommendations with educational "Why?" explanations
 */
import React, { useState } from 'react';

interface Recommendation {
    rec_id: string;
    column: string | null;
    rec_type: string;
    priority: string;
    title: string;
    description: string;
    why_explanation: string;
    suggested_action: string;
    alternative_actions: string[];
    impact_summary: string;
    code_snippet: string;
    learn_more_topics: string[];
    affected_rows: number;
    affected_percentage: number;
}

interface RecommendationCardProps {
    recommendation: Recommendation;
    onApply?: (rec: Recommendation) => void;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
    recommendation,
    onApply,
}) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [showCode, setShowCode] = useState(false);

    const getPriorityStyles = (priority: string) => {
        switch (priority.toLowerCase()) {
            case 'critical':
                return {
                    border: 'border-red-300',
                    bg: 'bg-red-50',
                    badge: 'bg-red-500 text-white',
                    icon: '🚨'
                };
            case 'high':
                return {
                    border: 'border-orange-300',
                    bg: 'bg-orange-50',
                    badge: 'bg-orange-500 text-white',
                    icon: '⚠️'
                };
            case 'medium':
                return {
                    border: 'border-yellow-300',
                    bg: 'bg-yellow-50',
                    badge: 'bg-yellow-500 text-white',
                    icon: '💡'
                };
            case 'low':
                return {
                    border: 'border-green-300',
                    bg: 'bg-green-50',
                    badge: 'bg-green-500 text-white',
                    icon: '✨'
                };
            default:
                return {
                    border: 'border-blue-300',
                    bg: 'bg-blue-50',
                    badge: 'bg-blue-500 text-white',
                    icon: 'ℹ️'
                };
        }
    };

    const styles = getPriorityStyles(recommendation.priority);

    return (
        <div className={`rounded-xl border ${styles.border} ${styles.bg} overflow-hidden`}>
            {/* Header */}
            <div className="p-4">
                <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${styles.badge}`}>
                                {styles.icon} {recommendation.priority.toUpperCase()}
                            </span>
                            <span className="text-xs text-slate-500">
                                {recommendation.rec_type.replace('_', ' ')}
                            </span>
                        </div>
                        <h3 className="font-semibold text-slate-900">{recommendation.title}</h3>
                        <p className="text-sm text-slate-600 mt-1">{recommendation.description}</p>

                        {recommendation.column && (
                            <div className="mt-2 inline-flex items-center gap-1 px-2 py-0.5 bg-slate-200 rounded text-xs text-slate-700">
                                <span>Column:</span>
                                <code className="font-mono">{recommendation.column}</code>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={() => onApply?.(recommendation)}
                        className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 
                       hover:bg-indigo-700 rounded-lg transition-colors shrink-0"
                    >
                        Fix Now
                    </button>
                </div>

                {/* Quick stats */}
                <div className="flex gap-4 mt-3 text-sm">
                    <span className="text-slate-500">
                        <strong className="text-slate-700">{recommendation.affected_rows.toLocaleString()}</strong> rows affected
                    </span>
                    <span className="text-slate-500">
                        ({recommendation.affected_percentage.toFixed(1)}% of data)
                    </span>
                </div>

                {/* Expand/Collapse */}
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="mt-3 text-sm font-medium text-indigo-600 hover:text-indigo-700 
                     flex items-center gap-1"
                >
                    {isExpanded ? '▲ Hide details' : '▼ Why this recommendation?'}
                </button>
            </div>

            {/* Expanded Details */}
            {isExpanded && (
                <div className="border-t border-slate-200 bg-white p-4 space-y-4">
                    {/* Why Explanation */}
                    <div>
                        <h4 className="text-sm font-semibold text-slate-900 mb-2 flex items-center gap-2">
                            🎓 Why This Matters
                        </h4>
                        <p className="text-sm text-slate-600 leading-relaxed">
                            {recommendation.why_explanation}
                        </p>
                    </div>

                    {/* Suggested Action */}
                    <div>
                        <h4 className="text-sm font-semibold text-slate-900 mb-2">
                            ✅ Suggested Action
                        </h4>
                        <p className="text-sm text-indigo-600 font-medium">
                            {recommendation.suggested_action}
                        </p>
                    </div>

                    {/* Alternative Actions */}
                    <div>
                        <h4 className="text-sm font-semibold text-slate-900 mb-2">
                            🔄 Alternative Approaches
                        </h4>
                        <ul className="text-sm text-slate-600 space-y-1">
                            {recommendation.alternative_actions.map((action, idx) => (
                                <li key={idx} className="flex items-start gap-2">
                                    <span className="text-slate-400">•</span>
                                    {action}
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Impact Summary */}
                    <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                        <h4 className="text-sm font-semibold text-green-800 mb-1">
                            📈 Expected Impact
                        </h4>
                        <p className="text-sm text-green-700">
                            {recommendation.impact_summary}
                        </p>
                    </div>

                    {/* Code Snippet */}
                    <div>
                        <button
                            onClick={() => setShowCode(!showCode)}
                            className="text-sm font-medium text-slate-600 hover:text-slate-900 
                         flex items-center gap-1"
                        >
                            {showCode ? '▲ Hide code' : '▼ Show code example'}
                        </button>

                        {showCode && (
                            <div className="mt-2 relative">
                                <pre className="p-3 bg-slate-900 text-green-400 text-xs rounded-lg overflow-x-auto">
                                    <code>{recommendation.code_snippet}</code>
                                </pre>
                                <button
                                    onClick={() => navigator.clipboard.writeText(recommendation.code_snippet)}
                                    className="absolute top-2 right-2 px-2 py-1 text-xs bg-slate-700 
                             text-slate-300 hover:bg-slate-600 rounded transition-colors"
                                >
                                    Copy
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Learn More Topics */}
                    <div>
                        <h4 className="text-sm font-semibold text-slate-900 mb-2">
                            📚 Learn More
                        </h4>
                        <div className="flex flex-wrap gap-2">
                            {recommendation.learn_more_topics.map((topic, idx) => (
                                <span
                                    key={idx}
                                    className="px-2 py-1 bg-slate-100 text-slate-600 text-xs rounded-full"
                                >
                                    {topic}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

/**
 * RecommendationsList Component
 * Displays a list of recommendations grouped by priority
 */
interface RecommendationsListProps {
    recommendations: Recommendation[];
    onApply?: (rec: Recommendation) => void;
}

export const RecommendationsList: React.FC<RecommendationsListProps> = ({
    recommendations,
    onApply,
}) => {
    if (!recommendations || recommendations.length === 0) {
        return (
            <div className="text-center py-12 bg-green-50 rounded-xl border border-green-200">
                <span className="text-4xl mb-4 block">🎉</span>
                <h3 className="text-lg font-semibold text-green-800 mb-2">
                    No Recommendations Needed
                </h3>
                <p className="text-green-600">
                    Your dataset looks great! No preprocessing issues detected.
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-slate-900">
                    Recommendations ({recommendations.length})
                </h2>
                <div className="flex gap-2 text-xs">
                    {['critical', 'high', 'medium', 'low'].map((priority) => {
                        const count = recommendations.filter(r => r.priority === priority).length;
                        if (count === 0) return null;
                        const colors: Record<string, string> = {
                            critical: 'bg-red-500',
                            high: 'bg-orange-500',
                            medium: 'bg-yellow-500',
                            low: 'bg-green-500',
                        };
                        return (
                            <span key={priority} className={`px-2 py-1 ${colors[priority]} text-white rounded-full`}>
                                {count} {priority}
                            </span>
                        );
                    })}
                </div>
            </div>

            {recommendations.map((rec) => (
                <RecommendationCard
                    key={rec.rec_id}
                    recommendation={rec}
                    onApply={onApply}
                />
            ))}
        </div>
    );
};

export default RecommendationCard;
