/**
 * QualityScoreCard Component
 * Displays quality scores with visual progress indicators
 */
import React from 'react';

interface QualityScoreCardProps {
    title: string;
    score: number;
    description?: string;
    icon?: React.ReactNode;
}

export const QualityScoreCard: React.FC<QualityScoreCardProps> = ({
    title,
    score,
    description,
    icon,
}) => {
    const getScoreColor = (value: number) => {
        if (value >= 80) return 'text-green-500';
        if (value >= 60) return 'text-yellow-500';
        return 'text-red-500';
    };

    const getProgressColor = (value: number) => {
        if (value >= 80) return 'bg-green-500';
        if (value >= 60) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    return (
        <div className="bg-card rounded-xl p-6 border border-border shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
                <div>
                    <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
                    <p className={`text-3xl font-bold ${getScoreColor(score)}`}>
                        {score.toFixed(0)}%
                    </p>
                </div>
                {icon && (
                    <div className="p-2 bg-muted rounded-lg">
                        {icon}
                    </div>
                )}
            </div>

            <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                    className={`h-full ${getProgressColor(score)} transition-all duration-500`}
                    style={{ width: `${Math.min(score, 100)}%` }}
                />
            </div>

            {description && (
                <p className="text-xs text-muted-foreground mt-2">{description}</p>
            )}
        </div>
    );
};

/**
 * IssueCard Component
 * Displays a single quality issue with severity indicator
 */
interface Issue {
    issue_id: string;
    issue_type: string;
    severity: string;
    column: string | null;
    description: string;
    affected_rows: number;
    affected_percentage: number;
    recommendation: string;
}

interface IssueCardProps {
    issue: Issue;
    onFix?: () => void;
}

export const IssueCard: React.FC<IssueCardProps> = ({ issue, onFix }) => {
    const getSeverityStyles = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'critical':
                return { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200' };
            case 'high':
                return { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200' };
            case 'medium':
                return { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-200' };
            case 'low':
                return { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200' };
            default:
                return { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-200' };
        }
    };

    const styles = getSeverityStyles(issue.severity);

    return (
        <div className={`p-4 rounded-lg border ${styles.border} ${styles.bg}`}>
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                        <span className={`text-xs font-semibold uppercase px-2 py-0.5 rounded ${styles.text} ${styles.bg}`}>
                            {issue.severity}
                        </span>
                        <span className="text-xs text-muted-foreground">
                            {issue.issue_type.replace(/_/g, ' ')}
                        </span>
                    </div>

                    <p className="text-sm font-medium text-foreground mb-1">
                        {issue.description}
                    </p>

                    {issue.column && (
                        <p className="text-xs text-muted-foreground">
                            Column: <span className="font-mono">{issue.column}</span>
                        </p>
                    )}

                    <p className="text-xs text-muted-foreground mt-2">
                        💡 {issue.recommendation}
                    </p>
                </div>

                {onFix && (
                    <button
                        onClick={onFix}
                        className="ml-4 px-3 py-1 text-sm font-medium text-primary bg-primary/10 rounded-lg hover:bg-primary/20 transition-colors"
                    >
                        Fix
                    </button>
                )}
            </div>
        </div>
    );
};

/**
 * StatCard Component
 * Simple stat display card
 */
interface StatCardProps {
    label: string;
    value: string | number;
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, icon, trend }) => {
    return (
        <div className="bg-card rounded-xl p-4 border border-border">
            <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">{label}</p>
                {icon}
            </div>
            <p className="text-2xl font-bold text-foreground mt-2">{value}</p>
            {trend && (
                <div className={`text-xs mt-1 ${trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : 'text-muted-foreground'}`}>
                    {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '−'} {trend}
                </div>
            )}
        </div>
    );
};

export default QualityScoreCard;
