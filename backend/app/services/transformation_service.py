"""
Transformation Engine Service for interactive data preprocessing.
Implements transformation operations with preview, apply, and undo capabilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from uuid import uuid4
import logging
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Types of transformation operations."""
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    IMPUTE_CONSTANT = "impute_constant"
    IMPUTE_FORWARD_FILL = "impute_forward_fill"
    IMPUTE_BACKWARD_FILL = "impute_backward_fill"
    REMOVE_OUTLIERS_ZSCORE = "remove_outliers_zscore"
    REMOVE_OUTLIERS_IQR = "remove_outliers_iqr"
    CAP_OUTLIERS_ZSCORE = "cap_outliers_zscore"
    CAP_OUTLIERS_IQR = "cap_outliers_iqr"
    ENCODE_ONEHOT = "encode_onehot"
    ENCODE_LABEL = "encode_label"
    CONVERT_DTYPE = "convert_dtype"
    DROP_COLUMN = "drop_column"
    DROP_ROWS = "drop_rows"
    FILTER_ROWS = "filter_rows"
    REMOVE_DUPLICATES = "remove_duplicates"
    RENAME_COLUMN = "rename_column"
    SCALE_STANDARD = "scale_standard"
    SCALE_MINMAX = "scale_minmax"


@dataclass
class TransformationParams:
    """Parameters for a transformation operation."""
    columns: List[str] = field(default_factory=list)
    constant_value: Optional[Any] = None
    threshold: float = 3.0  # For Z-score
    iqr_multiplier: float = 1.5  # For IQR
    target_dtype: Optional[str] = None
    filter_condition: Optional[str] = None
    new_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransformationResult:
    """Result of a transformation operation."""
    transformation_id: str
    transformation_type: TransformationType
    success: bool
    affected_rows: int
    affected_columns: List[str]
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    message: str
    timestamp: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['transformation_type'] = self.transformation_type.value
        return result


@dataclass
class TransformationPreview:
    """Preview of a transformation operation."""
    transformation_type: TransformationType
    can_apply: bool
    affected_rows: int
    affected_columns: List[str]
    before_sample: Dict[str, Any]
    after_sample: Dict[str, Any]
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    warnings: List[str]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['transformation_type'] = self.transformation_type.value
        return result


class TransformationService:
    """Service for data transformation operations with preview and history."""
    
    def __init__(self):
        """Initialize the transformation service."""
        self.logger = logging.getLogger(__name__)
        self._transformation_history: Dict[str, List[TransformationResult]] = {}
        self._df_snapshots: Dict[str, List[pd.DataFrame]] = {}
    
    def preview_transformation(
        self,
        df: pd.DataFrame,
        transformation_type: TransformationType,
        params: TransformationParams
    ) -> TransformationPreview:
        """
        Generate a preview of the transformation effect.
        
        Args:
            df: DataFrame to transform
            transformation_type: Type of transformation
            params: Transformation parameters
            
        Returns:
            TransformationPreview with before/after comparison
        """
        try:
            # Get before statistics
            before_stats = self._calculate_stats(df, params.columns)
            before_sample = df.head(10).to_dict(orient='records')
            
            # Apply transformation to a copy
            df_preview = df.copy()
            affected_rows, warnings = self._apply_transformation(
                df_preview, transformation_type, params, preview_only=True
            )
            
            # Get after statistics
            after_stats = self._calculate_stats(df_preview, params.columns)
            after_sample = df_preview.head(10).to_dict(orient='records')
            
            # Generate explanation
            explanation = self._generate_explanation(transformation_type, params, affected_rows, warnings)
            
            return TransformationPreview(
                transformation_type=transformation_type,
                can_apply=len(warnings) == 0 or all('warning' in w.lower() for w in warnings),
                affected_rows=affected_rows,
                affected_columns=params.columns,
                before_sample=before_sample,
                after_sample=after_sample,
                before_stats=before_stats,
                after_stats=after_stats,
                warnings=warnings,
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Error generating preview: {e}")
            return TransformationPreview(
                transformation_type=transformation_type,
                can_apply=False,
                affected_rows=0,
                affected_columns=params.columns,
                before_sample={},
                after_sample={},
                before_stats={},
                after_stats={},
                warnings=[f"Error: {str(e)}"],
                explanation=f"Preview failed: {str(e)}"
            )
    
    def apply_transformation(
        self,
        df: pd.DataFrame,
        transformation_type: TransformationType,
        params: TransformationParams,
        dataset_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, TransformationResult]:
        """
        Apply a transformation to the DataFrame.
        
        Args:
            df: DataFrame to transform
            transformation_type: Type of transformation
            params: Transformation parameters
            dataset_id: Optional dataset ID for history tracking
            
        Returns:
            Tuple of (transformed DataFrame, TransformationResult)
        """
        transformation_id = str(uuid4())
        
        try:
            # Store snapshot for undo
            if dataset_id:
                if dataset_id not in self._df_snapshots:
                    self._df_snapshots[dataset_id] = []
                self._df_snapshots[dataset_id].append(df.copy())
            
            # Get before statistics
            before_stats = self._calculate_stats(df, params.columns)
            
            # Apply transformation
            affected_rows, warnings = self._apply_transformation(df, transformation_type, params)
            
            # Get after statistics
            after_stats = self._calculate_stats(df, params.columns)
            
            result = TransformationResult(
                transformation_id=transformation_id,
                transformation_type=transformation_type,
                success=True,
                affected_rows=affected_rows,
                affected_columns=params.columns,
                before_stats=before_stats,
                after_stats=after_stats,
                message=f"Successfully applied {transformation_type.value}",
                timestamp=datetime.now().isoformat(),
                parameters=params.to_dict()
            )
            
            # Store in history
            if dataset_id:
                if dataset_id not in self._transformation_history:
                    self._transformation_history[dataset_id] = []
                self._transformation_history[dataset_id].append(result)
            
            return df, result
            
        except Exception as e:
            self.logger.error(f"Error applying transformation: {e}")
            return df, TransformationResult(
                transformation_id=transformation_id,
                transformation_type=transformation_type,
                success=False,
                affected_rows=0,
                affected_columns=params.columns,
                before_stats={},
                after_stats={},
                message=f"Failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                parameters=params.to_dict()
            )
    
    def _apply_transformation(
        self,
        df: pd.DataFrame,
        transformation_type: TransformationType,
        params: TransformationParams,
        preview_only: bool = False
    ) -> Tuple[int, List[str]]:
        """
        Internal method to apply transformation.
        
        Returns:
            Tuple of (affected_rows, warnings)
        """
        warnings = []
        affected_rows = 0
        
        if transformation_type == TransformationType.IMPUTE_MEAN:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    missing_count = df[col].isnull().sum()
                    df[col].fillna(df[col].mean(), inplace=True)
                    affected_rows += missing_count
                else:
                    warnings.append(f"Column {col} is not numeric, skipping mean imputation")
        
        elif transformation_type == TransformationType.IMPUTE_MEDIAN:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    missing_count = df[col].isnull().sum()
                    df[col].fillna(df[col].median(), inplace=True)
                    affected_rows += missing_count
                else:
                    warnings.append(f"Column {col} is not numeric, skipping median imputation")
        
        elif transformation_type == TransformationType.IMPUTE_MODE:
            for col in params.columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        affected_rows += missing_count
        
        elif transformation_type == TransformationType.IMPUTE_CONSTANT:
            for col in params.columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    df[col].fillna(params.constant_value, inplace=True)
                    affected_rows += missing_count
        
        elif transformation_type == TransformationType.IMPUTE_FORWARD_FILL:
            for col in params.columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    df[col].ffill(inplace=True)
                    affected_rows += missing_count
        
        elif transformation_type == TransformationType.IMPUTE_BACKWARD_FILL:
            for col in params.columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    df[col].bfill(inplace=True)
                    affected_rows += missing_count
        
        elif transformation_type == TransformationType.REMOVE_OUTLIERS_ZSCORE:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > params.threshold
                    affected_rows += outlier_mask.sum()
                    df.drop(df[outlier_mask].index, inplace=True)
                    df.reset_index(drop=True, inplace=True)
        
        elif transformation_type == TransformationType.REMOVE_OUTLIERS_IQR:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - params.iqr_multiplier * iqr
                    upper = q3 + params.iqr_multiplier * iqr
                    outlier_mask = (df[col] < lower) | (df[col] > upper)
                    affected_rows += outlier_mask.sum()
                    df.drop(df[outlier_mask].index, inplace=True)
                    df.reset_index(drop=True, inplace=True)
        
        elif transformation_type == TransformationType.CAP_OUTLIERS_ZSCORE:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    lower = mean_val - params.threshold * std_val
                    upper = mean_val + params.threshold * std_val
                    affected_rows += ((df[col] < lower) | (df[col] > upper)).sum()
                    df[col] = df[col].clip(lower=lower, upper=upper)
        
        elif transformation_type == TransformationType.CAP_OUTLIERS_IQR:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - params.iqr_multiplier * iqr
                    upper = q3 + params.iqr_multiplier * iqr
                    affected_rows += ((df[col] < lower) | (df[col] > upper)).sum()
                    df[col] = df[col].clip(lower=lower, upper=upper)
        
        elif transformation_type == TransformationType.ENCODE_ONEHOT:
            for col in params.columns:
                if col in df.columns:
                    unique_count = df[col].nunique()
                    if unique_count > 20:
                        warnings.append(f"Column {col} has {unique_count} unique values, one-hot encoding may create many columns")
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df.drop(columns=[col], inplace=True)
                    for dummy_col in dummies.columns:
                        df[dummy_col] = dummies[dummy_col]
                    affected_rows = len(df)
        
        elif transformation_type == TransformationType.ENCODE_LABEL:
            for col in params.columns:
                if col in df.columns:
                    categories = df[col].astype('category')
                    df[col] = categories.cat.codes
                    affected_rows = len(df)
        
        elif transformation_type == TransformationType.CONVERT_DTYPE:
            for col in params.columns:
                if col in df.columns and params.target_dtype:
                    try:
                        if params.target_dtype == 'int':
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        elif params.target_dtype == 'float':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif params.target_dtype == 'str':
                            df[col] = df[col].astype(str)
                        elif params.target_dtype == 'datetime':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif params.target_dtype == 'category':
                            df[col] = df[col].astype('category')
                        affected_rows = len(df)
                    except Exception as e:
                        warnings.append(f"Error converting {col} to {params.target_dtype}: {e}")
        
        elif transformation_type == TransformationType.DROP_COLUMN:
            for col in params.columns:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    affected_rows = len(df)
        
        elif transformation_type == TransformationType.REMOVE_DUPLICATES:
            original_len = len(df)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            affected_rows = original_len - len(df)
        
        elif transformation_type == TransformationType.RENAME_COLUMN:
            if len(params.columns) == 1 and params.new_name:
                col = params.columns[0]
                if col in df.columns:
                    df.rename(columns={col: params.new_name}, inplace=True)
                    affected_rows = len(df)
        
        elif transformation_type == TransformationType.SCALE_STANDARD:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[col] = (df[col] - mean_val) / std_val
                        affected_rows = len(df)
                    else:
                        warnings.append(f"Column {col} has zero variance, skipping standardization")
        
        elif transformation_type == TransformationType.SCALE_MINMAX:
            for col in params.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                        affected_rows = len(df)
                    else:
                        warnings.append(f"Column {col} has constant values, skipping min-max scaling")
        
        return affected_rows, warnings
    
    def _calculate_stats(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Calculate statistics for specified columns."""
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            col_stats = {
                "count": len(df[col]),
                "missing": df[col].isnull().sum(),
                "unique": df[col].nunique()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_stats.update({
                        "mean": round(float(non_null.mean()), 4),
                        "std": round(float(non_null.std()), 4),
                        "min": round(float(non_null.min()), 4),
                        "max": round(float(non_null.max()), 4),
                        "median": round(float(non_null.median()), 4)
                    })
            
            stats[col] = col_stats
        
        return stats
    
    def _generate_explanation(
        self,
        transformation_type: TransformationType,
        params: TransformationParams,
        affected_rows: int,
        warnings: List[str]
    ) -> str:
        """Generate human-readable explanation of the transformation."""
        explanations = {
            TransformationType.IMPUTE_MEAN: "Replaces missing values with the column mean. Best for normally distributed data without outliers.",
            TransformationType.IMPUTE_MEDIAN: "Replaces missing values with the column median. More robust to outliers than mean imputation.",
            TransformationType.IMPUTE_MODE: "Replaces missing values with the most frequent value. Suitable for categorical data.",
            TransformationType.IMPUTE_CONSTANT: f"Replaces missing values with the constant value: {params.constant_value}.",
            TransformationType.IMPUTE_FORWARD_FILL: "Propagates the last valid value forward to fill missing values. Good for time-series data.",
            TransformationType.IMPUTE_BACKWARD_FILL: "Uses the next valid value to fill missing values. Good for time-series data.",
            TransformationType.REMOVE_OUTLIERS_ZSCORE: f"Removes rows where Z-score exceeds {params.threshold}. Values beyond {params.threshold} standard deviations are considered outliers.",
            TransformationType.REMOVE_OUTLIERS_IQR: f"Removes rows beyond {params.iqr_multiplier}x the IQR from Q1/Q3. A common robust method for outlier detection.",
            TransformationType.CAP_OUTLIERS_ZSCORE: f"Caps values at {params.threshold} standard deviations from the mean. Preserves data while limiting extreme values.",
            TransformationType.CAP_OUTLIERS_IQR: f"Caps values at {params.iqr_multiplier}x the IQR from Q1/Q3. Preserves data while limiting extreme values.",
            TransformationType.ENCODE_ONEHOT: "Creates binary columns for each category. Suitable for nominal categorical variables.",
            TransformationType.ENCODE_LABEL: "Converts categories to integer codes. Suitable for ordinal data or tree-based models.",
            TransformationType.CONVERT_DTYPE: f"Converts column data type to {params.target_dtype}.",
            TransformationType.DROP_COLUMN: "Removes the specified column(s) from the dataset.",
            TransformationType.REMOVE_DUPLICATES: "Removes exact duplicate rows, keeping the first occurrence.",
            TransformationType.RENAME_COLUMN: f"Renames column to: {params.new_name}.",
            TransformationType.SCALE_STANDARD: "Standardizes values to have mean=0 and std=1. Required for many ML algorithms.",
            TransformationType.SCALE_MINMAX: "Scales values to range [0, 1]. Useful when you need bounded values."
        }
        
        base_explanation = explanations.get(transformation_type, "Applies the requested transformation.")
        
        return f"{base_explanation} This operation will affect {affected_rows} rows."
    
    def undo_transformation(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Undo the last transformation for a dataset.
        
        Args:
            dataset_id: Dataset ID to undo transformation for
            
        Returns:
            Previous DataFrame state or None if no history
        """
        if dataset_id in self._df_snapshots and self._df_snapshots[dataset_id]:
            return self._df_snapshots[dataset_id].pop()
        return None
    
    def get_transformation_history(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Get transformation history for a dataset.
        
        Args:
            dataset_id: Dataset ID to get history for
            
        Returns:
            List of transformation results
        """
        if dataset_id in self._transformation_history:
            return [t.to_dict() for t in self._transformation_history[dataset_id]]
        return []
    
    def clear_history(self, dataset_id: str) -> None:
        """Clear transformation history for a dataset."""
        if dataset_id in self._transformation_history:
            del self._transformation_history[dataset_id]
        if dataset_id in self._df_snapshots:
            del self._df_snapshots[dataset_id]


# Global instance
transformation_service = TransformationService()
