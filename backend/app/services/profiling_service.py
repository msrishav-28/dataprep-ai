"""
Data profiling service for comprehensive dataset analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from ydata_profiling import ProfileReport
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ProfilingService:
    """Service for comprehensive data profiling and statistical analysis."""
    
    def __init__(self):
        """Initialize the profiling service."""
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Generate comprehensive data profile using ydata-profiling and custom metrics.
        
        Args:
            df: DataFrame to profile
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dict containing comprehensive profiling results
        """
        try:
            # Generate ydata-profiling report
            profile = ProfileReport(
                df,
                title=f"Data Profile: {dataset_name}",
                explorative=True,
                minimal=False
            )
            
            # Extract key information from the profile
            profile_dict = profile.to_dict()
            
            # Generate custom platform-specific metrics
            custom_metrics = self._generate_custom_metrics(df)
            
            # Combine results
            comprehensive_profile = {
                "dataset_overview": {
                    "name": dataset_name,
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "memory_usage_bytes": df.memory_usage(deep=True).sum(),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                    "generated_at": datetime.now().isoformat()
                },
                "column_statistics": self._extract_column_statistics(df, profile_dict),
                "data_quality_summary": self._generate_quality_summary(df),
                "custom_metrics": custom_metrics,
                "ydata_profile": {
                    "variables": profile_dict.get("variables", {}),
                    "table": profile_dict.get("table", {}),
                    "correlations": profile_dict.get("correlations", {})
                }
            }
            
            return comprehensive_profile
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive profile: {e}")
            # Fallback to basic profiling if ydata-profiling fails
            return self._generate_basic_profile(df, dataset_name)
    
    def _generate_custom_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate platform-specific custom metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing custom metrics
        """
        metrics = {}
        
        try:
            # Data completeness metrics
            metrics["completeness"] = {
                "overall_completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
                "columns_with_missing": df.isnull().any().sum(),
                "rows_with_missing": df.isnull().any(axis=1).sum(),
                "completely_empty_columns": (df.isnull().all()).sum(),
                "completely_empty_rows": (df.isnull().all(axis=1)).sum()
            }
            
            # Data type distribution
            type_counts = df.dtypes.value_counts().to_dict()
            metrics["data_types"] = {str(k): int(v) for k, v in type_counts.items()}
            
            # Uniqueness metrics
            metrics["uniqueness"] = {}
            for col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                metrics["uniqueness"][col] = {
                    "unique_values": unique_count,
                    "uniqueness_ratio": round(unique_count / total_count if total_count > 0 else 0, 4),
                    "is_unique": unique_count == total_count,
                    "has_duplicates": unique_count < total_count
                }
            
            # Numerical column insights
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                metrics["numerical_insights"] = {}
                for col in numerical_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        metrics["numerical_insights"][col] = {
                            "zero_count": (col_data == 0).sum(),
                            "negative_count": (col_data < 0).sum(),
                            "positive_count": (col_data > 0).sum(),
                            "infinite_count": np.isinf(col_data).sum(),
                            "range": float(col_data.max() - col_data.min()) if len(col_data) > 1 else 0
                        }
            
            # Categorical column insights
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                metrics["categorical_insights"] = {}
                for col in categorical_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        value_counts = col_data.value_counts()
                        metrics["categorical_insights"][col] = {
                            "most_frequent_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                            "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                            "cardinality": len(value_counts),
                            "mode_frequency": round(value_counts.iloc[0] / len(col_data) * 100, 2) if len(value_counts) > 0 else 0
                        }
            
        except Exception as e:
            self.logger.error(f"Error generating custom metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _extract_column_statistics(self, df: pd.DataFrame, profile_dict: Dict) -> Dict[str, Any]:
        """
        Extract and organize column-level statistics.
        
        Args:
            df: DataFrame being profiled
            profile_dict: ydata-profiling results dictionary
            
        Returns:
            Dict containing organized column statistics
        """
        column_stats = {}
        
        try:
            variables = profile_dict.get("variables", {})
            
            for col in df.columns:
                col_info = variables.get(col, {})
                col_data = df[col]
                
                # Basic information
                stats = {
                    "name": col,
                    "dtype": str(col_data.dtype),
                    "count": len(col_data),
                    "missing_count": col_data.isnull().sum(),
                    "missing_percentage": round(col_data.isnull().sum() / len(col_data) * 100, 2),
                    "unique_count": col_data.nunique(),
                    "memory_usage": col_data.memory_usage(deep=True)
                }
                
                # Type-specific statistics
                if pd.api.types.is_numeric_dtype(col_data):
                    non_null_data = col_data.dropna()
                    if len(non_null_data) > 0:
                        stats.update({
                            "mean": float(non_null_data.mean()),
                            "median": float(non_null_data.median()),
                            "std": float(non_null_data.std()) if len(non_null_data) > 1 else 0,
                            "min": float(non_null_data.min()),
                            "max": float(non_null_data.max()),
                            "q25": float(non_null_data.quantile(0.25)),
                            "q75": float(non_null_data.quantile(0.75)),
                            "skewness": float(non_null_data.skew()) if len(non_null_data) > 1 else 0,
                            "kurtosis": float(non_null_data.kurtosis()) if len(non_null_data) > 1 else 0
                        })
                
                elif pd.api.types.is_string_dtype(col_data) or col_data.dtype == 'object':
                    non_null_data = col_data.dropna()
                    if len(non_null_data) > 0:
                        # String length statistics
                        str_lengths = non_null_data.astype(str).str.len()
                        stats.update({
                            "min_length": int(str_lengths.min()),
                            "max_length": int(str_lengths.max()),
                            "mean_length": round(float(str_lengths.mean()), 2),
                            "mode": str(non_null_data.mode().iloc[0]) if len(non_null_data.mode()) > 0 else None,
                            "top_values": non_null_data.value_counts().head(5).to_dict()
                        })
                
                column_stats[col] = stats
                
        except Exception as e:
            self.logger.error(f"Error extracting column statistics: {e}")
        
        return column_stats
    
    def _generate_quality_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data quality summary metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing quality summary
        """
        try:
            # Calculate duplicate rows
            duplicate_rows = df.duplicated().sum()
            
            # Calculate missing value patterns
            missing_patterns = df.isnull().sum()
            
            # Calculate data type consistency issues
            type_issues = 0
            for col in df.select_dtypes(include=['object']).columns:
                # Check if numeric values are stored as strings
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    type_issues += 1
                except (ValueError, TypeError):
                    pass
            
            quality_summary = {
                "overall_score": self._calculate_quality_score(df),
                "issues": {
                    "duplicate_rows": int(duplicate_rows),
                    "columns_with_missing": int((missing_patterns > 0).sum()),
                    "total_missing_values": int(missing_patterns.sum()),
                    "potential_type_issues": type_issues
                },
                "recommendations": self._generate_quality_recommendations(df)
            }
            
            return quality_summary
            
        except Exception as e:
            self.logger.error(f"Error generating quality summary: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Args:
            df: DataFrame to score
            
        Returns:
            Quality score as float
        """
        try:
            # Completeness score (40% weight)
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 40
            
            # Uniqueness score (30% weight) - penalize excessive duplicates
            duplicate_penalty = min(df.duplicated().sum() / len(df) * 30, 30)
            uniqueness = 30 - duplicate_penalty
            
            # Consistency score (30% weight) - basic type consistency check
            consistency = 30  # Start with full points
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    consistency -= 5  # Penalize potential type inconsistencies
                except (ValueError, TypeError):
                    pass
            
            consistency = max(consistency, 0)
            
            total_score = completeness + uniqueness + consistency
            return round(min(total_score, 100), 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _generate_quality_recommendations(self, df: pd.DataFrame) -> List[str]:
        """
        Generate actionable quality improvement recommendations.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            # Missing value recommendations
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                recommendations.append(f"Address missing values in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}")
            
            # Duplicate row recommendations
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                recommendations.append(f"Remove {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}% of data)")
            
            # Data type recommendations
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    recommendations.append(f"Convert column '{col}' to numeric type for better analysis")
                except (ValueError, TypeError):
                    pass
            
            # High cardinality recommendations
            for col in df.select_dtypes(include=['object']).columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.9:
                    recommendations.append(f"Column '{col}' has very high cardinality ({unique_ratio:.1%}) - consider if it should be an identifier")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _generate_basic_profile(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Generate basic profile when comprehensive profiling fails.
        
        Args:
            df: DataFrame to profile
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing basic profiling results
        """
        try:
            basic_profile = {
                "dataset_overview": {
                    "name": dataset_name,
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "memory_usage_bytes": df.memory_usage(deep=True).sum(),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                    "generated_at": datetime.now().isoformat()
                },
                "column_statistics": {},
                "data_quality_summary": self._generate_quality_summary(df),
                "custom_metrics": self._generate_custom_metrics(df),
                "ydata_profile": None
            }
            
            # Basic column statistics
            for col in df.columns:
                col_data = df[col]
                stats = {
                    "name": col,
                    "dtype": str(col_data.dtype),
                    "count": len(col_data),
                    "missing_count": col_data.isnull().sum(),
                    "missing_percentage": round(col_data.isnull().sum() / len(col_data) * 100, 2),
                    "unique_count": col_data.nunique()
                }
                
                if pd.api.types.is_numeric_dtype(col_data):
                    non_null_data = col_data.dropna()
                    if len(non_null_data) > 0:
                        stats.update({
                            "mean": float(non_null_data.mean()),
                            "median": float(non_null_data.median()),
                            "std": float(non_null_data.std()) if len(non_null_data) > 1 else 0,
                            "min": float(non_null_data.min()),
                            "max": float(non_null_data.max())
                        })
                
                basic_profile["column_statistics"][col] = stats
            
            return basic_profile
            
        except Exception as e:
            self.logger.error(f"Error generating basic profile: {e}")
            return {
                "error": str(e),
                "dataset_overview": {
                    "name": dataset_name,
                    "num_rows": 0,
                    "num_columns": 0,
                    "memory_usage_bytes": 0,
                    "memory_usage_mb": 0,
                    "generated_at": datetime.now().isoformat()
                }
            }
    
    def calculate_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate detailed memory usage information.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing memory usage details
        """
        try:
            memory_info = df.memory_usage(deep=True)
            
            return {
                "total_bytes": int(memory_info.sum()),
                "total_mb": round(memory_info.sum() / (1024 * 1024), 2),
                "index_bytes": int(memory_info.iloc[0]),
                "columns_bytes": {col: int(memory_info[col]) for col in df.columns},
                "largest_column": memory_info[1:].idxmax(),
                "largest_column_mb": round(memory_info[1:].max() / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating memory usage: {e}")
            return {"error": str(e)}


# Global instance
profiling_service = ProfilingService()