"""
Quality Assessment Service for comprehensive data quality analysis.
Implements detection algorithms for missing values, duplicates, outliers, and data type inconsistencies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    """Severity levels for data quality issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(str, Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    TYPE_INCONSISTENCY = "type_inconsistency"
    HIGH_CARDINALITY = "high_cardinality"
    LOW_VARIANCE = "low_variance"
    CONSTANT_COLUMN = "constant_column"
    MIXED_TYPES = "mixed_types"


@dataclass
class QualityIssue:
    """Represents a single data quality issue."""
    issue_id: str
    issue_type: IssueType
    severity: IssueSeverity
    column: Optional[str]
    description: str
    affected_rows: int
    affected_percentage: float
    recommendation: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['issue_type'] = self.issue_type.value
        result['severity'] = self.severity.value
        return result


class QualityAssessmentService:
    """Service for comprehensive data quality assessment."""
    
    def __init__(self):
        """Initialize the quality assessment service."""
        self.logger = logging.getLogger(__name__)
        self._issue_counter = 0
    
    def _generate_issue_id(self) -> str:
        """Generate unique issue ID."""
        self._issue_counter += 1
        return f"QI-{self._issue_counter:04d}"
    
    def assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment on a DataFrame.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dict containing complete quality assessment results
        """
        self._issue_counter = 0  # Reset counter for each assessment
        
        try:
            # Detect all quality issues
            issues = []
            issues.extend(self.detect_missing_values(df))
            issues.extend(self.detect_duplicates(df))
            issues.extend(self.detect_outliers(df))
            issues.extend(self.detect_type_inconsistencies(df))
            issues.extend(self.detect_additional_issues(df))
            
            # Sort issues by severity
            severity_order = {
                IssueSeverity.CRITICAL: 0,
                IssueSeverity.HIGH: 1,
                IssueSeverity.MEDIUM: 2,
                IssueSeverity.LOW: 3,
                IssueSeverity.INFO: 4
            }
            issues.sort(key=lambda x: severity_order[x.severity])
            
            # Calculate overall scores
            quality_scores = self._calculate_quality_scores(df, issues)
            
            # Generate recommendations
            prioritized_recommendations = self._generate_prioritized_recommendations(issues)
            
            return {
                "assessment_timestamp": datetime.now().isoformat(),
                "dataset_summary": {
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
                },
                "quality_scores": quality_scores,
                "issues": [issue.to_dict() for issue in issues],
                "issue_summary": {
                    "total_issues": len(issues),
                    "critical": sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
                    "high": sum(1 for i in issues if i.severity == IssueSeverity.HIGH),
                    "medium": sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM),
                    "low": sum(1 for i in issues if i.severity == IssueSeverity.LOW),
                    "info": sum(1 for i in issues if i.severity == IssueSeverity.INFO)
                },
                "prioritized_recommendations": prioritized_recommendations,
                "column_quality": self._assess_column_quality(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error during quality assessment: {e}")
            return {
                "error": str(e),
                "assessment_timestamp": datetime.now().isoformat(),
                "quality_scores": {"overall": 0},
                "issues": [],
                "prioritized_recommendations": []
            }
    
    def detect_missing_values(self, df: pd.DataFrame) -> List[QualityIssue]:
        """
        Detect missing values and calculate percentage missing for each column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of QualityIssue objects for missing value problems
        """
        issues = []
        
        try:
            total_rows = len(df)
            
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                
                if missing_count > 0:
                    missing_percentage = (missing_count / total_rows) * 100
                    
                    # Determine severity based on percentage
                    if missing_percentage >= 50:
                        severity = IssueSeverity.CRITICAL
                    elif missing_percentage >= 20:
                        severity = IssueSeverity.HIGH
                    elif missing_percentage >= 5:
                        severity = IssueSeverity.MEDIUM
                    else:
                        severity = IssueSeverity.LOW
                    
                    # Determine imputation strategy recommendation
                    col_dtype = df[col].dtype
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        if missing_percentage < 10:
                            rec = f"Impute with median (robust to outliers) or mean"
                        else:
                            rec = f"Consider using multiple imputation or model-based approaches"
                    elif pd.api.types.is_categorical_dtype(col_dtype) or col_dtype == 'object':
                        rec = f"Impute with mode or create 'Unknown' category"
                    else:
                        rec = f"Review column and choose appropriate imputation strategy"
                    
                    issues.append(QualityIssue(
                        issue_id=self._generate_issue_id(),
                        issue_type=IssueType.MISSING_VALUES,
                        severity=severity,
                        column=col,
                        description=f"Column '{col}' has {missing_count} missing values ({missing_percentage:.2f}%)",
                        affected_rows=missing_count,
                        affected_percentage=round(missing_percentage, 2),
                        recommendation=rec,
                        details={
                            "missing_count": missing_count,
                            "missing_percentage": round(missing_percentage, 2),
                            "column_dtype": str(col_dtype),
                            "non_null_count": total_rows - missing_count
                        }
                    ))
            
            # Check for rows with mostly missing values
            rows_with_many_missing = (df.isnull().sum(axis=1) > len(df.columns) * 0.5).sum()
            if rows_with_many_missing > 0:
                issues.append(QualityIssue(
                    issue_id=self._generate_issue_id(),
                    issue_type=IssueType.MISSING_VALUES,
                    severity=IssueSeverity.HIGH if rows_with_many_missing > total_rows * 0.1 else IssueSeverity.MEDIUM,
                    column=None,
                    description=f"{rows_with_many_missing} rows have more than 50% missing values",
                    affected_rows=rows_with_many_missing,
                    affected_percentage=round((rows_with_many_missing / total_rows) * 100, 2),
                    recommendation="Consider removing these mostly-empty rows or investigate data collection issues",
                    details={
                        "threshold": "50% missing",
                        "rows_affected": rows_with_many_missing
                    }
                ))
                
        except Exception as e:
            self.logger.error(f"Error detecting missing values: {e}")
        
        return issues
    
    def detect_duplicates(self, df: pd.DataFrame) -> List[QualityIssue]:
        """
        Identify duplicate rows and report the count.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of QualityIssue objects for duplicate problems
        """
        issues = []
        
        try:
            total_rows = len(df)
            
            # Exact duplicate rows
            exact_duplicates = df.duplicated().sum()
            if exact_duplicates > 0:
                dup_percentage = (exact_duplicates / total_rows) * 100
                
                if dup_percentage >= 10:
                    severity = IssueSeverity.HIGH
                elif dup_percentage >= 5:
                    severity = IssueSeverity.MEDIUM
                else:
                    severity = IssueSeverity.LOW
                
                issues.append(QualityIssue(
                    issue_id=self._generate_issue_id(),
                    issue_type=IssueType.DUPLICATES,
                    severity=severity,
                    column=None,
                    description=f"Found {exact_duplicates} exact duplicate rows ({dup_percentage:.2f}%)",
                    affected_rows=exact_duplicates,
                    affected_percentage=round(dup_percentage, 2),
                    recommendation="Remove duplicate rows using df.drop_duplicates() to ensure data integrity",
                    details={
                        "duplicate_count": exact_duplicates,
                        "unique_rows": total_rows - exact_duplicates,
                        "first_duplicate_indices": df[df.duplicated()].index.tolist()[:10]
                    }
                ))
            
            # Check for near-duplicates in key columns (if identifiable)
            potential_id_cols = [col for col in df.columns 
                               if 'id' in col.lower() or 'key' in col.lower() or 'code' in col.lower()]
            
            for col in potential_id_cols:
                dup_in_col = df[col].duplicated().sum()
                if dup_in_col > 0:
                    issues.append(QualityIssue(
                        issue_id=self._generate_issue_id(),
                        issue_type=IssueType.DUPLICATES,
                        severity=IssueSeverity.MEDIUM,
                        column=col,
                        description=f"Potential ID column '{col}' has {dup_in_col} duplicate values",
                        affected_rows=dup_in_col,
                        affected_percentage=round((dup_in_col / total_rows) * 100, 2),
                        recommendation=f"Verify if '{col}' should be unique; if so, investigate duplicates",
                        details={
                            "duplicate_values": df[df[col].duplicated()][col].head(10).tolist()
                        }
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error detecting duplicates: {e}")
        
        return issues
    
    def detect_outliers(self, df: pd.DataFrame) -> List[QualityIssue]:
        """
        Detect outliers using Z-score and IQR methods.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of QualityIssue objects for outlier problems
        """
        issues = []
        
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            total_rows = len(df)
            
            for col in numerical_cols:
                col_data = df[col].dropna()
                
                if len(col_data) < 10:  # Skip columns with too few values
                    continue
                
                # Z-score method (|z| > 3)
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                if std_val > 0:
                    z_scores = np.abs((col_data - mean_val) / std_val)
                    z_outliers = (z_scores > 3).sum()
                else:
                    z_outliers = 0
                
                # IQR method
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                else:
                    iqr_outliers = 0
                
                # Use the more conservative estimate
                outlier_count = max(z_outliers, iqr_outliers)
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(col_data)) * 100
                    
                    if outlier_percentage >= 10:
                        severity = IssueSeverity.HIGH
                    elif outlier_percentage >= 5:
                        severity = IssueSeverity.MEDIUM
                    else:
                        severity = IssueSeverity.LOW
                    
                    # Get actual outlier values for details
                    if std_val > 0:
                        outlier_mask = z_scores > 3
                        outlier_values = col_data[outlier_mask].head(10).tolist()
                    else:
                        outlier_values = []
                    
                    issues.append(QualityIssue(
                        issue_id=self._generate_issue_id(),
                        issue_type=IssueType.OUTLIERS,
                        severity=severity,
                        column=col,
                        description=f"Column '{col}' has {outlier_count} potential outliers ({outlier_percentage:.2f}%)",
                        affected_rows=outlier_count,
                        affected_percentage=round(outlier_percentage, 2),
                        recommendation="Review outliers manually; consider capping, removing, or transforming if appropriate",
                        details={
                            "z_score_outliers": int(z_outliers),
                            "iqr_outliers": int(iqr_outliers),
                            "mean": round(float(mean_val), 4),
                            "std": round(float(std_val), 4),
                            "q1": round(float(q1), 4),
                            "q3": round(float(q3), 4),
                            "iqr": round(float(iqr), 4),
                            "lower_bound": round(float(lower_bound), 4) if iqr > 0 else None,
                            "upper_bound": round(float(upper_bound), 4) if iqr > 0 else None,
                            "sample_outliers": [round(float(v), 4) for v in outlier_values]
                        }
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
        
        return issues
    
    def detect_type_inconsistencies(self, df: pd.DataFrame) -> List[QualityIssue]:
        """
        Identify inconsistent data types within columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of QualityIssue objects for type inconsistency problems
        """
        issues = []
        
        try:
            total_rows = len(df)
            
            for col in df.columns:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # Check if object column contains numeric values that should be converted
                if col_data.dtype == 'object':
                    numeric_convertible = 0
                    non_numeric = 0
                    
                    for val in col_data.head(1000):  # Sample for performance
                        try:
                            float(str(val).replace(',', '').replace(' ', ''))
                            numeric_convertible += 1
                        except (ValueError, TypeError):
                            non_numeric += 1
                    
                    sample_size = min(len(col_data), 1000)
                    
                    # If mostly numeric, suggest conversion
                    if numeric_convertible >= sample_size * 0.9:
                        issues.append(QualityIssue(
                            issue_id=self._generate_issue_id(),
                            issue_type=IssueType.TYPE_INCONSISTENCY,
                            severity=IssueSeverity.MEDIUM,
                            column=col,
                            description=f"Column '{col}' appears to contain numeric data stored as text",
                            affected_rows=len(col_data),
                            affected_percentage=100.0,
                            recommendation=f"Convert column to numeric type using pd.to_numeric()",
                            details={
                                "current_dtype": str(col_data.dtype),
                                "suggested_dtype": "float64 or int64",
                                "sample_values": col_data.head(5).tolist()
                            }
                        ))
                    
                    # If mixed types detected
                    elif 0 < numeric_convertible < sample_size * 0.9 and non_numeric > 0:
                        issues.append(QualityIssue(
                            issue_id=self._generate_issue_id(),
                            issue_type=IssueType.MIXED_TYPES,
                            severity=IssueSeverity.HIGH,
                            column=col,
                            description=f"Column '{col}' contains mixed data types (numeric and text)",
                            affected_rows=non_numeric,
                            affected_percentage=round((non_numeric / sample_size) * 100, 2),
                            recommendation="Investigate and standardize the data format, or split into separate columns",
                            details={
                                "numeric_values_pct": round((numeric_convertible / sample_size) * 100, 2),
                                "text_values_pct": round((non_numeric / sample_size) * 100, 2),
                                "sample_values": col_data.head(10).tolist()
                            }
                        ))
                
                # Check for potential date columns stored as objects
                if col_data.dtype == 'object':
                    date_patterns = 0
                    sample = col_data.head(100)
                    
                    for val in sample:
                        val_str = str(val)
                        # Simple heuristic for date-like patterns
                        if any(sep in val_str for sep in ['/', '-']) and any(c.isdigit() for c in val_str):
                            date_patterns += 1
                    
                    if date_patterns >= len(sample) * 0.8:
                        issues.append(QualityIssue(
                            issue_id=self._generate_issue_id(),
                            issue_type=IssueType.TYPE_INCONSISTENCY,
                            severity=IssueSeverity.LOW,
                            column=col,
                            description=f"Column '{col}' appears to contain dates stored as text",
                            affected_rows=len(col_data),
                            affected_percentage=100.0,
                            recommendation="Convert to datetime type using pd.to_datetime() for better analysis",
                            details={
                                "sample_values": col_data.head(5).tolist()
                            }
                        ))
                        
        except Exception as e:
            self.logger.error(f"Error detecting type inconsistencies: {e}")
        
        return issues
    
    def detect_additional_issues(self, df: pd.DataFrame) -> List[QualityIssue]:
        """
        Detect additional data quality issues like high cardinality and constant columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of QualityIssue objects
        """
        issues = []
        
        try:
            total_rows = len(df)
            
            for col in df.columns:
                col_data = df[col].dropna()
                unique_count = col_data.nunique()
                
                if len(col_data) == 0:
                    continue
                
                # Constant columns (only one unique value)
                if unique_count == 1:
                    issues.append(QualityIssue(
                        issue_id=self._generate_issue_id(),
                        issue_type=IssueType.CONSTANT_COLUMN,
                        severity=IssueSeverity.MEDIUM,
                        column=col,
                        description=f"Column '{col}' has only one unique value (constant)",
                        affected_rows=len(col_data),
                        affected_percentage=100.0,
                        recommendation="Consider removing this column as it provides no information for analysis",
                        details={
                            "constant_value": str(col_data.iloc[0])
                        }
                    ))
                
                # Very low variance for numeric columns
                elif pd.api.types.is_numeric_dtype(col_data):
                    if col_data.std() == 0 or (col_data.std() / (col_data.mean() + 1e-10)) < 0.01:
                        issues.append(QualityIssue(
                            issue_id=self._generate_issue_id(),
                            issue_type=IssueType.LOW_VARIANCE,
                            severity=IssueSeverity.INFO,
                            column=col,
                            description=f"Column '{col}' has very low variance",
                            affected_rows=len(col_data),
                            affected_percentage=100.0,
                            recommendation="Consider if this column is useful for analysis; low variance features may not be informative",
                            details={
                                "mean": round(float(col_data.mean()), 4),
                                "std": round(float(col_data.std()), 4),
                                "unique_count": unique_count
                            }
                        ))
                
                # High cardinality for categorical columns
                if col_data.dtype == 'object':
                    cardinality_ratio = unique_count / len(col_data)
                    
                    if cardinality_ratio > 0.9 and unique_count > 100:
                        issues.append(QualityIssue(
                            issue_id=self._generate_issue_id(),
                            issue_type=IssueType.HIGH_CARDINALITY,
                            severity=IssueSeverity.INFO,
                            column=col,
                            description=f"Column '{col}' has very high cardinality ({unique_count} unique values)",
                            affected_rows=unique_count,
                            affected_percentage=round(cardinality_ratio * 100, 2),
                            recommendation="This may be an ID column or free text; consider if categorical encoding is appropriate",
                            details={
                                "unique_count": unique_count,
                                "cardinality_ratio": round(cardinality_ratio, 4)
                            }
                        ))
                        
        except Exception as e:
            self.logger.error(f"Error detecting additional issues: {e}")
        
        return issues
    
    def _calculate_quality_scores(self, df: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
        """
        Calculate overall and component quality scores.
        
        Args:
            df: DataFrame being assessed
            issues: List of detected issues
            
        Returns:
            Dict containing quality scores
        """
        try:
            total_rows = len(df)
            total_cells = len(df) * len(df.columns)
            
            # Completeness score (0-100)
            missing_cells = df.isnull().sum().sum()
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 100
            
            # Uniqueness score (0-100)
            duplicate_rows = df.duplicated().sum()
            uniqueness_score = ((total_rows - duplicate_rows) / total_rows) * 100 if total_rows > 0 else 100
            
            # Consistency score based on type issues
            type_issues = [i for i in issues if i.issue_type in [IssueType.TYPE_INCONSISTENCY, IssueType.MIXED_TYPES]]
            consistency_score = max(0, 100 - len(type_issues) * 10)
            
            # Validity score based on outliers
            outlier_issues = [i for i in issues if i.issue_type == IssueType.OUTLIERS]
            total_outlier_pct = sum(i.affected_percentage for i in outlier_issues)
            validity_score = max(0, 100 - total_outlier_pct)
            
            # Overall score (weighted average)
            overall_score = (
                completeness_score * 0.35 +
                uniqueness_score * 0.25 +
                consistency_score * 0.25 +
                validity_score * 0.15
            )
            
            return {
                "overall": round(overall_score, 2),
                "completeness": round(completeness_score, 2),
                "uniqueness": round(uniqueness_score, 2),
                "consistency": round(consistency_score, 2),
                "validity": round(validity_score, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating quality scores: {e}")
            return {"overall": 0, "completeness": 0, "uniqueness": 0, "consistency": 0, "validity": 0}
    
    def _generate_prioritized_recommendations(self, issues: List[QualityIssue]) -> List[Dict[str, Any]]:
        """
        Generate prioritized list of recommendations based on detected issues.
        
        Args:
            issues: List of detected issues
            
        Returns:
            Prioritized list of recommendations
        """
        recommendations = []
        
        # Group issues by type
        issue_groups = {}
        for issue in issues:
            if issue.issue_type not in issue_groups:
                issue_groups[issue.issue_type] = []
            issue_groups[issue.issue_type].append(issue)
        
        priority_order = [
            (IssueType.MISSING_VALUES, "Handle Missing Values"),
            (IssueType.DUPLICATES, "Remove Duplicate Records"),
            (IssueType.TYPE_INCONSISTENCY, "Fix Data Type Issues"),
            (IssueType.MIXED_TYPES, "Standardize Mixed Type Columns"),
            (IssueType.OUTLIERS, "Address Outliers"),
            (IssueType.CONSTANT_COLUMN, "Remove Constant Columns"),
            (IssueType.HIGH_CARDINALITY, "Review High Cardinality Columns"),
            (IssueType.LOW_VARIANCE, "Consider Low Variance Features")
        ]
        
        priority = 1
        for issue_type, action_title in priority_order:
            if issue_type in issue_groups:
                group_issues = issue_groups[issue_type]
                columns_affected = [i.column for i in group_issues if i.column]
                
                recommendations.append({
                    "priority": priority,
                    "action": action_title,
                    "issue_type": issue_type.value,
                    "issue_count": len(group_issues),
                    "columns_affected": columns_affected[:10],  # Limit to 10
                    "highest_severity": max(i.severity.value for i in group_issues),
                    "specific_recommendations": [i.recommendation for i in group_issues[:5]]
                })
                priority += 1
        
        return recommendations
    
    def _assess_column_quality(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Assess quality for each individual column.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dict with quality metrics for each column
        """
        column_quality = {}
        
        try:
            for col in df.columns:
                col_data = df[col]
                non_null_data = col_data.dropna()
                
                total = len(col_data)
                missing = col_data.isnull().sum()
                unique = col_data.nunique()
                
                # Calculate column quality score
                completeness = ((total - missing) / total) * 100 if total > 0 else 0
                
                # Type quality (1 if consistent, lower if mixed)
                type_quality = 100
                if col_data.dtype == 'object' and len(non_null_data) > 0:
                    types_detected = set()
                    for val in non_null_data.head(100):
                        if isinstance(val, (int, float)):
                            types_detected.add('numeric')
                        else:
                            try:
                                float(str(val).replace(',', ''))
                                types_detected.add('numeric_string')
                            except:
                                types_detected.add('string')
                    
                    if len(types_detected) > 1:
                        type_quality = 60
                
                column_score = completeness * 0.6 + type_quality * 0.4
                
                column_quality[col] = {
                    "quality_score": round(column_score, 2),
                    "completeness": round(completeness, 2),
                    "type_quality": round(type_quality, 2),
                    "missing_count": missing,
                    "missing_percentage": round((missing / total) * 100, 2) if total > 0 else 0,
                    "unique_count": unique,
                    "dtype": str(col_data.dtype)
                }
                
        except Exception as e:
            self.logger.error(f"Error assessing column quality: {e}")
        
        return column_quality


# Global instance
quality_service = QualityAssessmentService()
