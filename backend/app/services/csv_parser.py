"""
CSV parsing service with pandas integration for data processing and analysis.
"""
import io
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, BinaryIO
from fastapi import HTTPException, status
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class CSVParserService:
    """Service for parsing CSV files and extracting metadata."""
    
    @staticmethod
    def parse_csv(file_content: BinaryIO, encoding: str, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse CSV file and extract dataset metadata.
        
        Args:
            file_content: File content as binary stream
            encoding: Character encoding to use for parsing
            filename: Original filename
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Parsed dataframe and metadata
            
        Raises:
            HTTPException: If parsing fails
        """
        try:
            file_content.seek(0)
            
            # Read CSV with pandas
            df = pd.read_csv(
                file_content,
                encoding=encoding,
                low_memory=False,  # Read entire file to infer dtypes properly
                na_values=['', 'NULL', 'null', 'None', 'none', 'NaN', 'nan', '#N/A', '#NULL!'],
                keep_default_na=True
            )
            
            logger.info(f"Successfully parsed CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Generate metadata
            metadata = CSVParserService._generate_metadata(df, filename)
            
            return df, metadata
            
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="CSV file is empty or contains no data"
            )
        except pd.errors.ParserError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse CSV file: {str(e)}"
            )
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Encoding error while parsing CSV: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing CSV {filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to parse CSV file due to unexpected error"
            )
    
    @staticmethod
    def _generate_metadata(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for the dataset.
        
        Args:
            df: Parsed pandas DataFrame
            filename: Original filename
            
        Returns:
            Dict[str, Any]: Dataset metadata
        """
        try:
            # Basic dataset information
            num_rows, num_columns = df.shape
            
            # Column information with data type inference
            column_info = {}
            for col in df.columns:
                col_data = df[col]
                
                # Infer data type
                dtype_info = CSVParserService._infer_column_type(col_data)
                
                # Calculate basic statistics
                stats = CSVParserService._calculate_column_stats(col_data, dtype_info['type'])
                
                column_info[col] = {
                    'dtype': str(col_data.dtype),
                    'inferred_type': dtype_info['type'],
                    'nullable': dtype_info['nullable'],
                    'missing_count': int(col_data.isnull().sum()),
                    'missing_percentage': float(col_data.isnull().sum() / len(col_data) * 100),
                    'unique_count': int(col_data.nunique()),
                    'stats': stats
                }
            
            # Overall dataset statistics
            total_missing = df.isnull().sum().sum()
            total_cells = num_rows * num_columns
            
            metadata = {
                'filename': filename,
                'num_rows': num_rows,
                'num_columns': num_columns,
                'total_missing_values': int(total_missing),
                'missing_percentage': float(total_missing / total_cells * 100) if total_cells > 0 else 0.0,
                'memory_usage_bytes': int(df.memory_usage(deep=True).sum()),
                'column_info': column_info,
                'duplicate_rows': int(df.duplicated().sum()),
                'data_types_summary': CSVParserService._summarize_data_types(column_info)
            }
            
            logger.info(f"Generated metadata for {filename}: {num_rows} rows, {num_columns} columns")
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            # Return basic metadata if detailed analysis fails
            return {
                'filename': filename,
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'error': f"Failed to generate detailed metadata: {str(e)}"
            }
    
    @staticmethod
    def _infer_column_type(series: pd.Series) -> Dict[str, Any]:
        """
        Infer the semantic data type of a column.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Dict[str, Any]: Type information
        """
        # Remove null values for type inference
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {'type': 'empty', 'nullable': True}
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return {'type': 'integer', 'nullable': series.isnull().any()}
            else:
                return {'type': 'float', 'nullable': series.isnull().any()}
        
        # Check if column is datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return {'type': 'datetime', 'nullable': series.isnull().any()}
        
        # Check if column is boolean
        if pd.api.types.is_bool_dtype(series):
            return {'type': 'boolean', 'nullable': series.isnull().any()}
        
        # For object/string columns, try to infer more specific types
        if pd.api.types.is_object_dtype(series):
            # Try to convert to numeric
            try:
                pd.to_numeric(non_null_series, errors='raise')
                return {'type': 'numeric_string', 'nullable': series.isnull().any()}
            except (ValueError, TypeError):
                pass
            
            # Try to convert to datetime
            try:
                pd.to_datetime(non_null_series, errors='raise', infer_datetime_format=True)
                return {'type': 'datetime_string', 'nullable': series.isnull().any()}
            except (ValueError, TypeError):
                pass
            
            # Check if it's categorical (low cardinality)
            unique_ratio = len(non_null_series.unique()) / len(non_null_series)
            if unique_ratio < 0.1 and len(non_null_series.unique()) < 50:
                return {'type': 'categorical', 'nullable': series.isnull().any()}
            
            return {'type': 'text', 'nullable': series.isnull().any()}
        
        # Default fallback
        return {'type': 'unknown', 'nullable': series.isnull().any()}
    
    @staticmethod
    def _calculate_column_stats(series: pd.Series, column_type: str) -> Dict[str, Any]:
        """
        Calculate appropriate statistics based on column type.
        
        Args:
            series: Pandas series to analyze
            column_type: Inferred column type
            
        Returns:
            Dict[str, Any]: Column statistics
        """
        stats = {}
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {'count': 0}
        
        try:
            # Common stats for all types
            stats['count'] = int(len(non_null_series))
            
            if column_type in ['integer', 'float', 'numeric_string']:
                # Numeric statistics
                if column_type == 'numeric_string':
                    numeric_series = pd.to_numeric(non_null_series, errors='coerce').dropna()
                else:
                    numeric_series = non_null_series
                
                if len(numeric_series) > 0:
                    stats.update({
                        'mean': float(numeric_series.mean()),
                        'median': float(numeric_series.median()),
                        'std': float(numeric_series.std()),
                        'min': float(numeric_series.min()),
                        'max': float(numeric_series.max()),
                        'q25': float(numeric_series.quantile(0.25)),
                        'q75': float(numeric_series.quantile(0.75))
                    })
            
            elif column_type in ['text', 'categorical']:
                # Text/categorical statistics
                stats.update({
                    'most_frequent': str(non_null_series.mode().iloc[0]) if len(non_null_series.mode()) > 0 else None,
                    'frequency_of_most': int(non_null_series.value_counts().iloc[0]) if len(non_null_series) > 0 else 0,
                    'avg_length': float(non_null_series.astype(str).str.len().mean()),
                    'max_length': int(non_null_series.astype(str).str.len().max()),
                    'min_length': int(non_null_series.astype(str).str.len().min())
                })
            
            elif column_type in ['datetime', 'datetime_string']:
                # Datetime statistics
                if column_type == 'datetime_string':
                    datetime_series = pd.to_datetime(non_null_series, errors='coerce').dropna()
                else:
                    datetime_series = non_null_series
                
                if len(datetime_series) > 0:
                    stats.update({
                        'min_date': str(datetime_series.min()),
                        'max_date': str(datetime_series.max()),
                        'date_range_days': int((datetime_series.max() - datetime_series.min()).days)
                    })
            
        except Exception as e:
            logger.warning(f"Error calculating stats for column type {column_type}: {e}")
            stats['error'] = str(e)
        
        return stats
    
    @staticmethod
    def _summarize_data_types(column_info: Dict[str, Any]) -> Dict[str, int]:
        """
        Summarize data types across all columns.
        
        Args:
            column_info: Column information dictionary
            
        Returns:
            Dict[str, int]: Count of each data type
        """
        type_counts = {}
        for col_data in column_info.values():
            col_type = col_data.get('inferred_type', 'unknown')
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        return type_counts
    
    @staticmethod
    def get_sample_data(df: pd.DataFrame, num_rows: int = None) -> Dict[str, Any]:
        """
        Get sample data from the dataframe for preview.
        
        Args:
            df: Pandas DataFrame
            num_rows: Number of rows to sample (default from settings)
            
        Returns:
            Dict[str, Any]: Sample data and metadata
        """
        if num_rows is None:
            num_rows = min(settings.MAX_ROWS_PREVIEW, len(df))
        
        try:
            # Get sample rows
            sample_df = df.head(num_rows)
            
            # Convert to JSON-serializable format
            sample_data = {
                'columns': list(df.columns),
                'data': sample_df.fillna('').to_dict('records'),  # Fill NaN with empty string for JSON
                'total_rows': len(df),
                'sample_rows': len(sample_df),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            return {
                'error': f"Failed to generate sample data: {str(e)}",
                'columns': list(df.columns) if hasattr(df, 'columns') else [],
                'total_rows': len(df) if hasattr(df, '__len__') else 0
            }


# Global instance
csv_parser = CSVParserService()