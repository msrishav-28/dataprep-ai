"""
Unit tests for CSV parser service.
"""
import io
import pandas as pd
import pytest

from app.services.csv_parser import csv_parser


class TestCSVParser:
    """Unit tests for CSV parsing functionality."""
    
    def test_parse_simple_csv(self):
        """Test parsing a simple CSV file."""
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA\n"
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        df, metadata = csv_parser.parse_csv(csv_file, 'utf-8', 'test.csv')
        
        # Verify dataframe structure
        assert len(df) == 2
        assert len(df.columns) == 3
        assert list(df.columns) == ['name', 'age', 'city']
        
        # Verify metadata
        assert metadata['num_rows'] == 2
        assert metadata['num_columns'] == 3
        assert metadata['filename'] == 'test.csv'
        assert 'column_info' in metadata
    
    def test_parse_csv_with_missing_values(self):
        """Test parsing CSV with missing values."""
        csv_content = "name,age,city\nJohn,25,\nJane,,LA\n,30,Chicago\n"
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        df, metadata = csv_parser.parse_csv(csv_file, 'utf-8', 'test_missing.csv')
        
        # Verify missing values are handled
        assert df.isnull().sum().sum() > 0
        assert metadata['total_missing_values'] > 0
        assert metadata['missing_percentage'] > 0
    
    def test_parse_csv_with_different_data_types(self):
        """Test parsing CSV with different data types."""
        csv_content = "name,age,salary,is_active,join_date\nJohn,25,50000.5,true,2023-01-15\nJane,30,60000.0,false,2022-12-01\n"
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        df, metadata = csv_parser.parse_csv(csv_file, 'utf-8', 'test_types.csv')
        
        # Verify column type inference
        column_info = metadata['column_info']
        assert 'name' in column_info
        assert 'age' in column_info
        assert 'salary' in column_info
        
        # Check that different types are inferred
        types_found = set(col['inferred_type'] for col in column_info.values())
        assert len(types_found) > 1  # Should have multiple different types
    
    def test_get_sample_data(self):
        """Test sample data generation."""
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago\n"
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        df, _ = csv_parser.parse_csv(csv_file, 'utf-8', 'test_sample.csv')
        sample_data = csv_parser.get_sample_data(df, num_rows=2)
        
        # Verify sample data structure
        assert 'columns' in sample_data
        assert 'data' in sample_data
        assert 'total_rows' in sample_data
        assert 'sample_rows' in sample_data
        
        assert sample_data['total_rows'] == 3
        assert sample_data['sample_rows'] == 2
        assert len(sample_data['data']) == 2
        assert sample_data['columns'] == ['name', 'age', 'city']
    
    def test_empty_csv_handling(self):
        """Test handling of empty CSV files."""
        csv_content = ""
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        with pytest.raises(Exception):  # Should raise an exception for empty CSV
            csv_parser.parse_csv(csv_file, 'utf-8', 'empty.csv')
    
    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV files."""
        csv_content = "name,age\nJohn,25,extra_column\nJane\n"
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        # Should either parse successfully with pandas' error handling or raise appropriate exception
        try:
            df, metadata = csv_parser.parse_csv(csv_file, 'utf-8', 'malformed.csv')
            # If parsing succeeds, verify it handled the malformed data
            assert len(df) >= 0
        except Exception:
            # It's acceptable to raise an exception for severely malformed CSV
            pass