"""
Recommendation Engine for intelligent preprocessing suggestions.
Provides contextual recommendations with educational explanations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RecommendationType(str, Enum):
    """Types of preprocessing recommendations."""
    IMPUTATION = "imputation"
    OUTLIER_TREATMENT = "outlier_treatment"
    ENCODING = "encoding"
    SCALING = "scaling"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_CLEANING = "data_cleaning"
    TYPE_CONVERSION = "type_conversion"
    DUPLICATE_HANDLING = "duplicate_handling"


@dataclass
class Recommendation:
    """A preprocessing recommendation with educational context."""
    rec_id: str
    column: Optional[str]
    rec_type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    why_explanation: str  # Educational "Why?" explanation
    suggested_action: str
    alternative_actions: List[str]
    impact_summary: str
    code_snippet: str
    learn_more_topics: List[str]
    affected_rows: int = 0
    affected_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecommendationEngine:
    """Engine for generating intelligent preprocessing recommendations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        quality_assessment: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate all recommendations for a dataset.
        
        Args:
            df: The DataFrame to analyze
            quality_assessment: Optional quality assessment results
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        rec_id = 0
        
        # Missing value recommendations
        for rec in self._recommend_missing_values(df):
            rec_id += 1
            rec.rec_id = f"rec_{rec_id}"
            recommendations.append(rec.to_dict())
        
        # Outlier recommendations
        for rec in self._recommend_outliers(df):
            rec_id += 1
            rec.rec_id = f"rec_{rec_id}"
            recommendations.append(rec.to_dict())
        
        # Encoding recommendations
        for rec in self._recommend_encoding(df):
            rec_id += 1
            rec.rec_id = f"rec_{rec_id}"
            recommendations.append(rec.to_dict())
        
        # Scaling recommendations
        for rec in self._recommend_scaling(df):
            rec_id += 1
            rec.rec_id = f"rec_{rec_id}"
            recommendations.append(rec.to_dict())
        
        # Duplicate recommendations
        for rec in self._recommend_duplicates(df):
            rec_id += 1
            rec.rec_id = f"rec_{rec_id}"
            recommendations.append(rec.to_dict())
        
        # Data type recommendations
        for rec in self._recommend_type_conversions(df):
            rec_id += 1
            rec.rec_id = f"rec_{rec_id}"
            recommendations.append(rec.to_dict())
        
        # Sort by priority
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
            RecommendationPriority.INFO: 4
        }
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
        
        return recommendations
    
    def _recommend_missing_values(self, df: pd.DataFrame) -> List[Recommendation]:
        """Generate recommendations for missing values."""
        recommendations = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count == 0:
                continue
            
            dtype = df[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            
            # Determine priority
            if missing_pct > 50:
                priority = RecommendationPriority.CRITICAL
            elif missing_pct > 20:
                priority = RecommendationPriority.HIGH
            elif missing_pct > 5:
                priority = RecommendationPriority.MEDIUM
            else:
                priority = RecommendationPriority.LOW
            
            # Generate recommendation based on data type
            if is_numeric:
                # Check distribution for numeric columns
                skewness = df[col].skew() if df[col].notna().sum() > 2 else 0
                
                if abs(skewness) > 1:
                    suggested = "median"
                    why = (
                        f"The '{col}' column has a skewed distribution (skewness: {skewness:.2f}). "
                        f"Median imputation is recommended because it's robust to outliers and "
                        f"preserves the central tendency better than mean for skewed data. "
                        f"The mean would be pulled toward extreme values, distorting your data."
                    )
                else:
                    suggested = "mean"
                    why = (
                        f"The '{col}' column has a relatively symmetric distribution. "
                        f"Mean imputation is suitable here as it preserves the overall average "
                        f"and is appropriate when data is normally distributed. For skewed data, "
                        f"median would be preferred."
                    )
                
                code = f"""# Impute missing values in '{col}' with {suggested}
df['{col}'].fillna(df['{col}'].{suggested}(), inplace=True)"""
                
                alternatives = [
                    f"Use median imputation (robust to outliers)",
                    f"Use mean imputation (preserves average)",
                    f"Use KNN imputation (considers similar rows)",
                    f"Drop rows with missing values (if data is abundant)"
                ]
            else:
                suggested = "mode"
                why = (
                    f"The '{col}' column is categorical/text. Mode imputation fills missing "
                    f"values with the most frequent value, which is the standard approach for "
                    f"categorical data. This preserves the distribution of categories. "
                    f"If missingness is informative, consider creating a separate 'Unknown' category."
                )
                
                code = f"""# Impute missing values in '{col}' with mode
mode_value = df['{col}'].mode()[0]
df['{col}'].fillna(mode_value, inplace=True)"""
                
                alternatives = [
                    f"Use mode (most frequent value)",
                    f"Create 'Unknown' category (if missingness is meaningful)",
                    f"Forward/backward fill (for time-ordered data)",
                    f"Drop rows with missing values"
                ]
            
            recommendations.append(Recommendation(
                rec_id="",
                column=col,
                rec_type=RecommendationType.IMPUTATION,
                priority=priority,
                title=f"Handle {missing_count:,} missing values in '{col}'",
                description=f"{missing_pct:.1f}% of values are missing in this column.",
                why_explanation=why,
                suggested_action=f"Use {suggested} imputation",
                alternative_actions=alternatives,
                impact_summary=f"Will fill {missing_count:,} missing values, restoring data completeness.",
                code_snippet=code,
                learn_more_topics=["Missing Data Patterns", "Imputation Strategies", "MCAR vs MNAR"],
                affected_rows=missing_count,
                affected_percentage=missing_pct
            ))
        
        return recommendations
    
    def _recommend_outliers(self, df: pd.DataFrame) -> List[Recommendation]:
        """Generate recommendations for outliers."""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            # IQR method
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((data < lower) | (data > upper)).sum()
            outlier_pct = (outliers / len(data)) * 100
            
            if outliers == 0:
                continue
            
            # Determine priority
            if outlier_pct > 10:
                priority = RecommendationPriority.HIGH
            elif outlier_pct > 5:
                priority = RecommendationPriority.MEDIUM
            else:
                priority = RecommendationPriority.LOW
            
            why = (
                f"Found {outliers:,} outliers ({outlier_pct:.1f}%) in '{col}' using the IQR method. "
                f"Values below {lower:.2f} or above {upper:.2f} are considered outliers. "
                f"Outliers can significantly impact statistical analyses and ML models, especially "
                f"distance-based algorithms (KNN, SVM) and regression models. "
                f"However, some outliers may be valid extreme values - domain knowledge is key."
            )
            
            code = f"""# Cap outliers in '{col}' using IQR method
Q1 = df['{col}'].quantile(0.25)
Q3 = df['{col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['{col}'] = df['{col}'].clip(lower=lower_bound, upper=upper_bound)"""
            
            recommendations.append(Recommendation(
                rec_id="",
                column=col,
                rec_type=RecommendationType.OUTLIER_TREATMENT,
                priority=priority,
                title=f"Handle {outliers:,} outliers in '{col}'",
                description=f"{outlier_pct:.1f}% of values are statistical outliers.",
                why_explanation=why,
                suggested_action="Cap outliers using IQR method",
                alternative_actions=[
                    "Cap outliers at percentile bounds (Winsorization)",
                    "Remove outlier rows (if data is abundant)",
                    "Apply log transformation (for right-skewed data)",
                    "Keep outliers (if they represent valid extreme cases)"
                ],
                impact_summary=f"Will cap {outliers:,} extreme values to distribution bounds.",
                code_snippet=code,
                learn_more_topics=["IQR Method", "Z-Score Method", "Winsorization", "Robust Statistics"],
                affected_rows=outliers,
                affected_percentage=outlier_pct
            ))
        
        return recommendations
    
    def _recommend_encoding(self, df: pd.DataFrame) -> List[Recommendation]:
        """Generate recommendations for categorical encoding."""
        recommendations = []
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            n_unique = df[col].nunique()
            
            if n_unique > 50:
                priority = RecommendationPriority.MEDIUM
                suggested = "target encoding"
                why = (
                    f"The '{col}' column has {n_unique} unique categories, which is high. "
                    f"One-hot encoding would create {n_unique} new columns, leading to the "
                    f"'curse of dimensionality' and sparse matrices. Target encoding or "
                    f"frequency encoding is preferred for high-cardinality features."
                )
                code = f"""# Frequency encoding for high-cardinality column '{col}'
freq = df['{col}'].value_counts(normalize=True)
df['{col}_encoded'] = df['{col}'].map(freq)"""
                alternatives = [
                    "Frequency encoding (encode by occurrence rate)",
                    "Target encoding (encode by target mean)",
                    "Group rare categories into 'Other'",
                    "Use embedding (for deep learning)"
                ]
            elif n_unique <= 5:
                priority = RecommendationPriority.LOW
                suggested = "one-hot encoding"
                why = (
                    f"The '{col}' column has only {n_unique} unique categories, making it "
                    f"ideal for one-hot encoding. This creates {n_unique} binary columns, "
                    f"which is manageable and preserves the categorical nature without "
                    f"implying ordinal relationships between categories."
                )
                code = f"""# One-hot encode '{col}'
df = pd.get_dummies(df, columns=['{col}'], prefix='{col}', drop_first=True)"""
                alternatives = [
                    "One-hot encoding (creates binary columns)",
                    "Label encoding (if ordinal relationship exists)",
                    "Binary encoding (compromise between one-hot and label)"
                ]
            else:
                priority = RecommendationPriority.LOW
                suggested = "one-hot encoding"
                why = (
                    f"The '{col}' column has {n_unique} unique categories. One-hot encoding "
                    f"is the standard approach for nominal (unordered) categorical variables. "
                    f"It creates explicit features for each category, helping ML models learn "
                    f"category-specific patterns without assuming ordinal relationships."
                )
                code = f"""# One-hot encode '{col}'
df = pd.get_dummies(df, columns=['{col}'], prefix='{col}', drop_first=True)"""
                alternatives = [
                    "One-hot encoding (standard for nominal data)",
                    "Label encoding (if categories have order)",
                    "Frequency encoding (for moderately high cardinality)"
                ]
            
            recommendations.append(Recommendation(
                rec_id="",
                column=col,
                rec_type=RecommendationType.ENCODING,
                priority=priority,
                title=f"Encode categorical column '{col}'",
                description=f"Contains {n_unique} unique categories requiring encoding for ML.",
                why_explanation=why,
                suggested_action=f"Use {suggested}",
                alternative_actions=alternatives,
                impact_summary=f"Transforms text categories into numeric features for ML models.",
                code_snippet=code,
                learn_more_topics=["Categorical Encoding", "One-Hot vs Label", "Target Encoding"],
                affected_rows=len(df),
                affected_percentage=100.0
            ))
        
        return recommendations
    
    def _recommend_scaling(self, df: pd.DataFrame) -> List[Recommendation]:
        """Generate recommendations for feature scaling."""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return recommendations
        
        # Check for scale differences
        scales = {}
        for col in numeric_cols:
            col_range = df[col].max() - df[col].min()
            scales[col] = col_range
        
        if len(scales) == 0:
            return recommendations
        
        max_scale = max(scales.values())
        min_scale = min(scales.values()) if min(scales.values()) > 0 else 1
        
        if max_scale / min_scale > 10:
            # Significant scale differences
            why = (
                f"Numeric features have very different scales (ratio: {max_scale/min_scale:.0f}x). "
                f"Many ML algorithms (KNN, SVM, Neural Networks, PCA) are sensitive to feature scales. "
                f"Features with larger ranges will dominate distance calculations. "
                f"Scaling ensures all features contribute equally to the model."
            )
            
            recommendations.append(Recommendation(
                rec_id="",
                column=None,
                rec_type=RecommendationType.SCALING,
                priority=RecommendationPriority.MEDIUM,
                title="Scale numeric features to similar ranges",
                description="Features have significantly different scales which may affect ML performance.",
                why_explanation=why,
                suggested_action="Apply StandardScaler (z-score normalization)",
                alternative_actions=[
                    "StandardScaler (mean=0, std=1) - best for normally distributed data",
                    "MinMaxScaler (0 to 1) - preserves shape, good for bounded data",
                    "RobustScaler - uses median/IQR, robust to outliers",
                    "No scaling - if using tree-based models (Random Forest, XGBoost)"
                ],
                impact_summary=f"Normalizes {len(numeric_cols)} numeric features to similar scales.",
                code_snippet="""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])""",
                learn_more_topics=["Feature Scaling", "StandardScaler vs MinMaxScaler", "When to Scale"],
                affected_rows=len(df),
                affected_percentage=100.0
            ))
        
        return recommendations
    
    def _recommend_duplicates(self, df: pd.DataFrame) -> List[Recommendation]:
        """Generate recommendations for duplicate handling."""
        recommendations = []
        
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df)) * 100
        
        if dup_count == 0:
            return recommendations
        
        priority = RecommendationPriority.HIGH if dup_pct > 5 else RecommendationPriority.MEDIUM
        
        why = (
            f"Found {dup_count:,} duplicate rows ({dup_pct:.1f}% of data). "
            f"Duplicate rows can bias model training, inflate performance metrics, and "
            f"lead to data leakage if duplicates appear in both train and test sets. "
            f"However, ensure duplicates are truly errors and not valid repeated observations."
        )
        
        recommendations.append(Recommendation(
            rec_id="",
            column=None,
            rec_type=RecommendationType.DUPLICATE_HANDLING,
            priority=priority,
            title=f"Remove {dup_count:,} duplicate rows",
            description=f"{dup_pct:.1f}% of rows are exact duplicates.",
            why_explanation=why,
            suggested_action="Remove duplicate rows",
            alternative_actions=[
                "Remove all duplicates (keep first occurrence)",
                "Remove duplicates, keep last occurrence",
                "Keep duplicates (if valid repeated measurements)",
                "Remove near-duplicates using fuzzy matching"
            ],
            impact_summary=f"Reduces dataset from {len(df):,} to {len(df)-dup_count:,} rows.",
            code_snippet="df.drop_duplicates(inplace=True)\ndf.reset_index(drop=True, inplace=True)",
            learn_more_topics=["Duplicate Detection", "Data Leakage", "Train-Test Contamination"],
            affected_rows=dup_count,
            affected_percentage=dup_pct
        ))
        
        return recommendations
    
    def _recommend_type_conversions(self, df: pd.DataFrame) -> List[Recommendation]:
        """Generate recommendations for data type conversions."""
        recommendations = []
        
        for col in df.columns:
            # Check for numeric strings
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                valid_numeric_pct = numeric_converted.notna().sum() / len(df) * 100
                
                if valid_numeric_pct > 90:
                    recommendations.append(Recommendation(
                        rec_id="",
                        column=col,
                        rec_type=RecommendationType.TYPE_CONVERSION,
                        priority=RecommendationPriority.MEDIUM,
                        title=f"Convert '{col}' to numeric type",
                        description=f"{valid_numeric_pct:.1f}% of values can be converted to numbers.",
                        why_explanation=(
                            f"The '{col}' column is stored as text but contains numeric values. "
                            f"Converting to numeric type enables mathematical operations, "
                            f"statistical analysis, and proper handling by ML algorithms. "
                            f"Text-stored numbers are treated as categories, not quantities."
                        ),
                        suggested_action="Convert to numeric (int/float)",
                        alternative_actions=[
                            "Convert to integer (if no decimals)",
                            "Convert to float (if decimals present)",
                            "Keep as string (if it's an ID or code)"
                        ],
                        impact_summary="Enables proper numeric operations and ML compatibility.",
                        code_snippet=f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')",
                        learn_more_topics=["Data Types", "Type Conversion", "Pandas dtypes"],
                        affected_rows=len(df),
                        affected_percentage=100.0
                    ))
        
        return recommendations


# Global instance
recommendation_engine = RecommendationEngine()
