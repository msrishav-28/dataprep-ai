"""
Report Generation Service for creating comprehensive HTML reports.
Generates self-contained reports with visualizations, statistics, and transformation history.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64
import logging

logger = logging.getLogger(__name__)


class ReportGeneratorService:
    """Service for generating comprehensive HTML reports."""
    
    def __init__(self):
        """Initialize the report generator service."""
        self.logger = logging.getLogger(__name__)
    
    def generate_html_report(
        self,
        df: pd.DataFrame,
        profile_data: Optional[Dict[str, Any]] = None,
        quality_assessment: Optional[Dict[str, Any]] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
        visualizations: Optional[Dict[str, Any]] = None,
        dataset_name: str = "Dataset"
    ) -> str:
        """
        Generate a comprehensive HTML report.
        
        Args:
            df: The DataFrame (current state)
            profile_data: Profiling results
            quality_assessment: Quality assessment results
            transformations: List of applied transformations
            visualizations: Visualization configurations
            dataset_name: Name of the dataset
            
        Returns:
            Self-contained HTML string
        """
        try:
            html_parts = []
            
            # HTML Header with embedded styles
            html_parts.append(self._generate_html_header(dataset_name))
            
            # Report Header
            html_parts.append(self._generate_report_header(dataset_name, df))
            
            # Dataset Overview Section
            html_parts.append(self._generate_overview_section(df, profile_data))
            
            # Quality Assessment Section
            if quality_assessment:
                html_parts.append(self._generate_quality_section(quality_assessment))
            
            # Column Statistics Section
            html_parts.append(self._generate_column_statistics_section(df, profile_data))
            
            # Visualizations Section
            if visualizations:
                html_parts.append(self._generate_visualizations_section(visualizations))
            
            # Transformation History Section
            if transformations:
                html_parts.append(self._generate_transformation_section(transformations))
            
            # Footer
            html_parts.append(self._generate_footer())
            
            return '\n'.join(html_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"
    
    def _generate_html_header(self, title: str) -> str:
        """Generate HTML header with embedded CSS."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Report: {title}</title>
    <style>
        :root {{
            --primary-color: #4f46e5;
            --secondary-color: #6366f1;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --text-color: #1f2937;
            --bg-color: #f9fafb;
            --card-bg: #ffffff;
            --border-color: #e5e7eb;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
        }}
        
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            opacity: 0.9;
        }}
        
        .section {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .section h2 {{
            color: var(--primary-color);
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .stat-card .label {{
            color: #6b7280;
            font-size: 0.875rem;
        }}
        
        .quality-score {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .score-circle {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        }}
        
        .score-high {{ background: var(--success-color); }}
        .score-medium {{ background: var(--warning-color); }}
        .score-low {{ background: var(--danger-color); }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: #f3f4f6;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f9fafb;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .badge-critical {{ background: #fef2f2; color: #991b1b; }}
        .badge-high {{ background: #fff7ed; color: #c2410c; }}
        .badge-medium {{ background: #fffbeb; color: #b45309; }}
        .badge-low {{ background: #f0fdf4; color: #166534; }}
        .badge-info {{ background: #eff6ff; color: #1d4ed8; }}
        
        .transformation-list {{
            list-style: none;
        }}
        
        .transformation-item {{
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem;
            border-left: 3px solid var(--primary-color);
            margin-bottom: 0.5rem;
            background: #f9fafb;
        }}
        
        .transformation-item .step {{
            background: var(--primary-color);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ max-width: 100%; }}
        }}
    </style>
</head>
<body>
<div class="container">
'''
    
    def _generate_report_header(self, dataset_name: str, df: pd.DataFrame) -> str:
        """Generate report header section."""
        return f'''
<div class="header">
    <h1>📊 Data Analysis Report</h1>
    <p>Dataset: <strong>{dataset_name}</strong></p>
    <p>Generated: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
</div>
'''
    
    def _generate_overview_section(self, df: pd.DataFrame, profile_data: Optional[Dict]) -> str:
        """Generate dataset overview section."""
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicates = df.duplicated().sum()
        
        return f'''
<div class="section">
    <h2>📋 Dataset Overview</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{len(df):,}</div>
            <div class="label">Total Rows</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(df.columns)}</div>
            <div class="label">Total Columns</div>
        </div>
        <div class="stat-card">
            <div class="value">{memory_mb:.2f} MB</div>
            <div class="label">Memory Usage</div>
        </div>
        <div class="stat-card">
            <div class="value">{missing_pct:.1f}%</div>
            <div class="label">Missing Values</div>
        </div>
        <div class="stat-card">
            <div class="value">{duplicates:,}</div>
            <div class="label">Duplicate Rows</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(df.select_dtypes(include=[np.number]).columns)}</div>
            <div class="label">Numeric Columns</div>
        </div>
    </div>
</div>
'''
    
    def _generate_quality_section(self, quality_assessment: Dict[str, Any]) -> str:
        """Generate quality assessment section."""
        scores = quality_assessment.get('quality_scores', {})
        overall_score = scores.get('overall', 0)
        issues = quality_assessment.get('issues', [])
        
        score_class = 'score-high' if overall_score >= 80 else ('score-medium' if overall_score >= 60 else 'score-low')
        
        html = f'''
<div class="section">
    <h2>✅ Data Quality Assessment</h2>
    
    <div class="quality-score">
        <div class="score-circle {score_class}">
            {overall_score:.0f}
        </div>
        <div>
            <strong>Overall Quality Score</strong>
            <p>Completeness: {scores.get('completeness', 0):.1f}% | 
               Uniqueness: {scores.get('uniqueness', 0):.1f}% | 
               Consistency: {scores.get('consistency', 0):.1f}%</p>
        </div>
    </div>
'''
        
        if issues:
            html += '''
    <h3 style="margin-top: 1.5rem; margin-bottom: 0.5rem;">Detected Issues</h3>
    <table>
        <thead>
            <tr>
                <th>Severity</th>
                <th>Type</th>
                <th>Column</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
'''
            for issue in issues[:10]:  # Limit to 10 issues
                severity = issue.get('severity', 'info')
                html += f'''
            <tr>
                <td><span class="badge badge-{severity}">{severity.upper()}</span></td>
                <td>{issue.get('issue_type', 'unknown')}</td>
                <td>{issue.get('column', 'N/A')}</td>
                <td>{issue.get('description', '')}</td>
            </tr>
'''
            html += '''
        </tbody>
    </table>
'''
        
        html += '</div>'
        return html
    
    def _generate_column_statistics_section(self, df: pd.DataFrame, profile_data: Optional[Dict]) -> str:
        """Generate column statistics section."""
        html = '''
<div class="section">
    <h2>📊 Column Statistics</h2>
    <table>
        <thead>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Non-Null</th>
                <th>Missing %</th>
                <th>Unique</th>
                <th>Sample Values</th>
            </tr>
        </thead>
        <tbody>
'''
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique = df[col].nunique()
            
            # Get sample values
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ', '.join([str(v)[:20] for v in sample_values])
            
            html += f'''
            <tr>
                <td><strong>{col}</strong></td>
                <td>{dtype}</td>
                <td>{non_null:,}</td>
                <td>{missing_pct:.1f}%</td>
                <td>{unique:,}</td>
                <td>{sample_str}...</td>
            </tr>
'''
        
        html += '''
        </tbody>
    </table>
</div>
'''
        return html
    
    def _generate_visualizations_section(self, visualizations: Dict[str, Any]) -> str:
        """Generate visualizations section placeholder."""
        charts = visualizations.get('charts', [])
        
        html = '''
<div class="section">
    <h2>📈 Visualizations</h2>
    <p>This report includes {0} generated visualizations. Interactive charts are available in the platform interface.</p>
    <ul>
'''.format(len(charts))
        
        for chart in charts[:10]:
            html += f'        <li>{chart.get("title", "Untitled Chart")}</li>\n'
        
        html += '''
    </ul>
</div>
'''
        return html
    
    def _generate_transformation_section(self, transformations: List[Dict[str, Any]]) -> str:
        """Generate transformation history section."""
        html = '''
<div class="section">
    <h2>🔄 Transformation History</h2>
    <ul class="transformation-list">
'''
        
        for i, transform in enumerate(transformations, 1):
            transform_type = transform.get('transformation_type', 'unknown')
            params = transform.get('parameters', {})
            columns = params.get('columns', [])
            timestamp = transform.get('timestamp', '')
            
            html += f'''
        <li class="transformation-item">
            <span class="step">{i}</span>
            <div>
                <strong>{transform_type.replace("_", " ").title()}</strong>
                <p>Columns: {', '.join(columns) if columns else 'All'}</p>
                <small>{timestamp}</small>
            </div>
        </li>
'''
        
        html += '''
    </ul>
</div>
'''
        return html
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f'''
<div class="footer">
    <p>Generated by DataPrep AI | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>This report is self-contained and can be viewed offline.</p>
</div>
</div>
</body>
</html>
'''


# Global instance
report_generator = ReportGeneratorService()
