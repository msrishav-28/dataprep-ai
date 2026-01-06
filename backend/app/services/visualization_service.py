"""
Visualization Generation Service for interactive data visualizations.
Implements chart generation using Plotly for distributions, correlations, and quality heatmaps.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Types of charts supported by the visualization service."""
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    BAR_CHART = "bar_chart"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation_matrix"
    MISSING_VALUE_HEATMAP = "missing_value_heatmap"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    VIOLIN = "violin"


@dataclass
class ChartConfiguration:
    """Configuration for a chart visualization."""
    chart_id: str
    chart_type: ChartType
    title: str
    column: Optional[str]
    data: Dict[str, Any]
    layout: Dict[str, Any]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['chart_type'] = self.chart_type.value
        return result
    
    def to_json(self) -> str:
        """Convert to Plotly JSON format."""
        return json.dumps({
            "data": self.data,
            "layout": self.layout,
            "config": self.config
        })


class VisualizationService:
    """Service for generating interactive data visualizations."""
    
    def __init__(self):
        """Initialize the visualization service."""
        self.logger = logging.getLogger(__name__)
        self._chart_counter = 0
        
        # Default Plotly configuration
        self.default_config = {
            "responsive": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False
        }
        
        # Default color scheme
        self.color_palette = px.colors.qualitative.Set2
    
    def _generate_chart_id(self) -> str:
        """Generate unique chart ID."""
        self._chart_counter += 1
        return f"chart-{self._chart_counter:04d}"
    
    def generate_all_visualizations(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Generate all appropriate visualizations for a dataset.
        
        Args:
            df: DataFrame to visualize
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing all generated chart configurations
        """
        self._chart_counter = 0  # Reset counter
        
        try:
            charts = []
            
            # Generate distribution plots for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_cols[:10]:  # Limit to prevent too many charts
                chart = self.generate_histogram(df, col)
                if chart:
                    charts.append(chart)
                
                box_chart = self.generate_box_plot(df, col)
                if box_chart:
                    charts.append(box_chart)
            
            # Generate count plots for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols[:10]:
                unique_count = df[col].nunique()
                if 1 < unique_count <= 20:  # Reasonable cardinality for bar chart
                    chart = self.generate_bar_chart(df, col)
                    if chart:
                        charts.append(chart)
            
            # Generate correlation heatmap for numerical variables
            if len(numerical_cols) >= 2:
                corr_chart = self.generate_correlation_heatmap(df)
                if corr_chart:
                    charts.append(corr_chart)
            
            # Generate missing value heatmap
            if df.isnull().any().any():
                missing_chart = self.generate_missing_value_heatmap(df)
                if missing_chart:
                    charts.append(missing_chart)
            
            return {
                "generated_at": datetime.now().isoformat(),
                "dataset_name": dataset_name,
                "total_charts": len(charts),
                "charts": [c.to_dict() for c in charts],
                "numerical_columns": numerical_cols,
                "categorical_columns": categorical_cols
            }
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
                "charts": []
            }
    
    def generate_histogram(self, df: pd.DataFrame, column: str, bins: int = 30) -> Optional[ChartConfiguration]:
        """
        Generate a histogram for a numerical column.
        
        Args:
            df: DataFrame containing the data
            column: Column name to visualize
            bins: Number of bins for the histogram
            
        Returns:
            ChartConfiguration for the histogram
        """
        try:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                return None
            
            fig = px.histogram(
                df, 
                x=column, 
                nbins=bins,
                title=f"Distribution of {column}",
                color_discrete_sequence=[self.color_palette[0]]
            )
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count",
                showlegend=False,
                template="plotly_white"
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.HISTOGRAM,
                title=f"Distribution of {column}",
                column=column,
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating histogram for {column}: {e}")
            return None
    
    def generate_box_plot(self, df: pd.DataFrame, column: str) -> Optional[ChartConfiguration]:
        """
        Generate a box plot for outlier visualization.
        
        Args:
            df: DataFrame containing the data
            column: Column name to visualize
            
        Returns:
            ChartConfiguration for the box plot
        """
        try:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                return None
            
            fig = px.box(
                df,
                y=column,
                title=f"Box Plot: {column}",
                color_discrete_sequence=[self.color_palette[1]]
            )
            
            fig.update_layout(
                yaxis_title=column,
                showlegend=False,
                template="plotly_white"
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.BOX_PLOT,
                title=f"Box Plot: {column}",
                column=column,
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating box plot for {column}: {e}")
            return None
    
    def generate_bar_chart(self, df: pd.DataFrame, column: str, top_n: int = 20) -> Optional[ChartConfiguration]:
        """
        Generate a bar chart (count plot) for categorical variables.
        
        Args:
            df: DataFrame containing the data
            column: Column name to visualize
            top_n: Maximum number of categories to show
            
        Returns:
            ChartConfiguration for the bar chart
        """
        try:
            value_counts = df[column].value_counts().head(top_n)
            
            if len(value_counts) == 0:
                return None
            
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Value Counts: {column}",
                color_discrete_sequence=[self.color_palette[2]]
            )
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count",
                showlegend=False,
                template="plotly_white",
                xaxis_tickangle=-45
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.BAR_CHART,
                title=f"Value Counts: {column}",
                column=column,
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating bar chart for {column}: {e}")
            return None
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> Optional[ChartConfiguration]:
        """
        Generate a correlation heatmap for numerical variables.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            ChartConfiguration for the correlation heatmap
        """
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return None
            
            # Limit to first 15 columns to maintain readability
            cols_to_use = numerical_cols[:15]
            corr_matrix = df[cols_to_use].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="",
                yaxis_title=""
            )
            
            # Add correlation values as text
            fig.update_traces(
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.CORRELATION_MATRIX,
                title="Correlation Matrix",
                column=None,
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {e}")
            return None
    
    def generate_missing_value_heatmap(self, df: pd.DataFrame) -> Optional[ChartConfiguration]:
        """
        Generate a heatmap showing patterns of missing values.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            ChartConfiguration for the missing value heatmap
        """
        try:
            # Create binary matrix of missing values
            missing_matrix = df.isnull().astype(int)
            
            # Sample rows if too many
            if len(missing_matrix) > 100:
                missing_matrix = missing_matrix.sample(n=100, random_state=42)
            
            # Only include columns with missing values
            cols_with_missing = missing_matrix.columns[missing_matrix.any()]
            
            if len(cols_with_missing) == 0:
                return None
            
            missing_matrix = missing_matrix[cols_with_missing]
            
            fig = px.imshow(
                missing_matrix.values,
                x=cols_with_missing.tolist(),
                y=[str(i) for i in missing_matrix.index],
                title="Missing Value Patterns",
                color_continuous_scale=[[0, "lightgray"], [1, "crimson"]],
                aspect="auto"
            )
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Columns",
                yaxis_title="Rows (sample)",
                coloraxis_showscale=False
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.MISSING_VALUE_HEATMAP,
                title="Missing Value Patterns",
                column=None,
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating missing value heatmap: {e}")
            return None
    
    def generate_scatter_plot(
        self, 
        df: pd.DataFrame, 
        x_column: str, 
        y_column: str,
        color_column: Optional[str] = None
    ) -> Optional[ChartConfiguration]:
        """
        Generate a scatter plot for two numerical variables.
        
        Args:
            df: DataFrame containing the data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            color_column: Optional column for color encoding
            
        Returns:
            ChartConfiguration for the scatter plot
        """
        try:
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                title=f"{x_column} vs {y_column}",
                color_discrete_sequence=self.color_palette
            )
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.SCATTER,
                title=f"{x_column} vs {y_column}",
                column=f"{x_column},{y_column}",
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating scatter plot: {e}")
            return None
    
    def generate_violin_plot(self, df: pd.DataFrame, column: str, group_column: Optional[str] = None) -> Optional[ChartConfiguration]:
        """
        Generate a violin plot for distribution visualization.
        
        Args:
            df: DataFrame containing the data
            column: Column name to visualize
            group_column: Optional column for grouping
            
        Returns:
            ChartConfiguration for the violin plot
        """
        try:
            fig = px.violin(
                df,
                y=column,
                x=group_column,
                title=f"Violin Plot: {column}",
                color_discrete_sequence=[self.color_palette[4]]
            )
            
            fig.update_layout(
                template="plotly_white",
                yaxis_title=column
            )
            
            fig_dict = fig.to_dict()
            
            return ChartConfiguration(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.VIOLIN,
                title=f"Violin Plot: {column}",
                column=column,
                data=fig_dict.get("data", []),
                layout=fig_dict.get("layout", {}),
                config=self.default_config
            )
            
        except Exception as e:
            self.logger.error(f"Error generating violin plot for {column}: {e}")
            return None
    
    def export_chart_to_html(self, chart: ChartConfiguration, include_plotly_js: bool = True) -> str:
        """
        Export a chart configuration to standalone HTML.
        
        Args:
            chart: ChartConfiguration to export
            include_plotly_js: Whether to include Plotly.js library
            
        Returns:
            HTML string containing the chart
        """
        try:
            fig = go.Figure(data=chart.data, layout=chart.layout)
            return fig.to_html(
                include_plotlyjs=include_plotly_js,
                full_html=True,
                config=chart.config
            )
        except Exception as e:
            self.logger.error(f"Error exporting chart to HTML: {e}")
            return f"<html><body><p>Error generating chart: {e}</p></body></html>"
    
    def export_chart_to_image(self, chart: ChartConfiguration, format: str = "png", width: int = 800, height: int = 600) -> Optional[bytes]:
        """
        Export a chart configuration to image bytes.
        
        Args:
            chart: ChartConfiguration to export
            format: Image format (png, jpeg, svg, pdf)
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image bytes or None if export fails
        """
        try:
            fig = go.Figure(data=chart.data, layout=chart.layout)
            return fig.to_image(format=format, width=width, height=height)
        except Exception as e:
            self.logger.error(f"Error exporting chart to image: {e}")
            return None


# Global instance
visualization_service = VisualizationService()
