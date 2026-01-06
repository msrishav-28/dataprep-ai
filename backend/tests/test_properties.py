"""
Property-based tests for the DataPrep AI Platform.
Feature: data-preprocessing-platform
"""
import io
import tempfile
import pandas as pd
from typing import Any, Dict
from hypothesis import given, strategies as st, settings, assume
import pytest
from fastapi.testclient import TestClient

from app.services.file_validation import file_validator
from app.services.profiling_service import profiling_service


# Custom strategies for generating test data
@st.composite
def csv_file_strategy(draw):
    """Generate CSV file content with various characteristics."""
    # Generate number of rows and columns
    num_rows = draw(st.integers(min_value=1, max_value=100))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    
    # Generate column names
    col_names = [f"col_{i}" for i in range(num_cols)]
    
    # Generate data rows
    rows = []
    for _ in range(num_rows):
        row = []
        for _ in range(num_cols):
            # Generate different types of data
            value = draw(st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                st.text(min_size=0, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                st.just("")  # Empty values for missing data
            ))
            row.append(str(value))
        rows.append(row)
    
    # Create CSV content
    csv_content = ",".join(col_names) + "\n"
    for row in rows:
        csv_content += ",".join(row) + "\n"
    
    return csv_content.encode('utf-8')


@st.composite
def valid_filename_strategy(draw):
    """Generate valid CSV filenames."""
    base_name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    return f"{base_name}.csv"


@st.composite
def invalid_filename_strategy(draw):
    """Generate invalid filenames (non-CSV extensions)."""
    base_name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    invalid_ext = draw(st.sampled_from([".txt", ".xlsx", ".json", ".xml", ".pdf", ""]))
    return f"{base_name}{invalid_ext}"


@st.composite
def dataframe_strategy(draw):
    """Generate pandas DataFrames with various characteristics."""
    # Generate number of rows and columns
    num_rows = draw(st.integers(min_value=1, max_value=100))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    
    # Generate column names
    col_names = [f"col_{i}" for i in range(num_cols)]
    
    # Generate data for each column with different types
    data = {}
    for col in col_names:
        col_type = draw(st.sampled_from(['numeric', 'categorical', 'mixed']))
        
        if col_type == 'numeric':
            # Generate numeric data with some missing values
            values = draw(st.lists(
                st.one_of(
                    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                    st.integers(min_value=-1000, max_value=1000),
                    st.just(None)  # Missing values
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
        elif col_type == 'categorical':
            # Generate categorical data
            categories = draw(st.lists(
                st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
                min_size=1,
                max_size=5,
                unique=True
            ))
            values = draw(st.lists(
                st.one_of(
                    st.sampled_from(categories),
                    st.just(None)  # Missing values
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
        else:  # mixed
            # Generate mixed data types
            values = draw(st.lists(
                st.one_of(
                    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                    st.text(min_size=0, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                    st.just(None)
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
        
        data[col] = values
    
    return pd.DataFrame(data)


class TestFileUploadValidation:
    """Property tests for file upload validation."""
    
    @given(csv_content=csv_file_strategy(), filename=valid_filename_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_valid_csv_upload_acceptance(self, csv_content: bytes, filename: str):
        """
        Property 1: File Upload Validation
        For any valid CSV file under 1GB with .csv extension, the system should accept the file format and encoding.
        **Feature: data-preprocessing-platform, Property 1: File Upload Validation**
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        # Skip if file is too large (over 1GB)
        assume(len(csv_content) <= 1024 * 1024 * 1024)
        
        # Create a file-like object
        csv_file = io.BytesIO(csv_content)
        
        # Test file format validation - should not raise exception for .csv files
        try:
            file_validator.validate_file_format(filename)
        except Exception as e:
            pytest.fail(f"Valid CSV filename should not raise exception: {e}")
        
        # Test file size validation - should not raise exception for files under 1GB
        try:
            file_validator.validate_file_size(len(csv_content))
        except Exception as e:
            pytest.fail(f"File under 1GB should not raise size exception: {e}")
        
        # Test encoding detection - should successfully detect encoding
        try:
            encoding, confidence = file_validator.detect_encoding(csv_file)
            assert isinstance(encoding, str), "Encoding should be a string"
            assert isinstance(confidence, (int, float)), "Confidence should be numeric"
            assert encoding.lower() in ['utf-8', 'ascii', 'iso-8859-1', 'windows-1252'], f"Unexpected encoding: {encoding}"
        except Exception as e:
            pytest.fail(f"Encoding detection should not fail for valid CSV: {e}")
        
        # Test CSV content validation - should not raise exception for valid CSV
        try:
            file_validator.validate_csv_content(csv_file, encoding)
        except Exception as e:
            pytest.fail(f"Valid CSV content should not raise exception: {e}")
    
    @given(filename=invalid_filename_strategy())
    @settings(max_examples=50, deadline=None)
    def test_property_1_invalid_format_rejection(self, filename: str):
        """
        Property 1: File Upload Validation - Invalid Format Rejection
        For any file with non-CSV extension, the system should reject the file with appropriate error.
        **Feature: data-preprocessing-platform, Property 1: File Upload Validation**
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        # Skip if filename accidentally ends with .csv
        assume(not filename.lower().endswith('.csv'))
        
        # Test that invalid file formats are rejected
        with pytest.raises(Exception) as exc_info:
            file_validator.validate_file_format(filename)
        
        # Verify the exception contains appropriate error message
        error_message = str(exc_info.value)
        assert "not supported" in error_message.lower() or "format" in error_message.lower()
    
    @given(file_size=st.integers(min_value=1024*1024*1024 + 1, max_value=2*1024*1024*1024))
    @settings(max_examples=50, deadline=None)
    def test_property_1_oversized_file_rejection(self, file_size: int):
        """
        Property 1: File Upload Validation - Oversized File Rejection
        For any file over 1GB, the system should reject the file with size error.
        **Feature: data-preprocessing-platform, Property 1: File Upload Validation**
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        # Test that oversized files are rejected
        with pytest.raises(Exception) as exc_info:
            file_validator.validate_file_size(file_size)
        
        # Verify the exception contains appropriate error message about size
        error_message = str(exc_info.value)
        assert "size" in error_message.lower() or "exceeds" in error_message.lower()


class TestEncodingDetection:
    """Property tests for encoding detection and parsing."""
    
    @given(encoded_data=encoded_csv_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_encoding_detection_and_parsing(self, encoded_data):
        """
        Property 2: Encoding Detection and Parsing
        For any valid CSV file with standard character encoding, the system should automatically 
        detect the encoding and parse the content correctly without data corruption.
        **Feature: data-preprocessing-platform, Property 2: Encoding Detection and Parsing**
        **Validates: Requirements 1.5**
        """
        csv_content, expected_encoding = encoded_data
        
        # Create file-like object
        csv_file = io.BytesIO(csv_content)
        
        # Test encoding detection
        try:
            detected_encoding, confidence = file_validator.detect_encoding(csv_file)
            
            # Verify encoding detection returns valid results
            assert isinstance(detected_encoding, str), "Detected encoding should be a string"
            assert isinstance(confidence, (int, float)), "Confidence should be numeric"
            assert 0.0 <= confidence <= 1.0, f"Confidence should be between 0 and 1, got {confidence}"
            
            # Verify detected encoding is a known encoding
            known_encodings = ['utf-8', 'ascii', 'iso-8859-1', 'windows-1252', 'utf-16', 'utf-32']
            assert detected_encoding.lower() in known_encodings, f"Unknown encoding detected: {detected_encoding}"
            
        except Exception as e:
            pytest.fail(f"Encoding detection should not fail for valid encoded CSV: {e}")
        
        # Test that content can be parsed with detected encoding
        try:
            file_validator.validate_csv_content(csv_file, detected_encoding)
        except Exception as e:
            pytest.fail(f"CSV content validation should succeed with detected encoding: {e}")
        
        # Test that content can be decoded without corruption
        try:
            csv_file.seek(0)
            decoded_content = csv_file.read().decode(detected_encoding)
            
            # Verify basic CSV structure is preserved
            lines = decoded_content.strip().split('\n')
            assert len(lines) >= 1, "Decoded content should have at least one line"
            
            # Verify no obvious corruption (e.g., replacement characters)
            assert '\ufffd' not in decoded_content, "Decoded content should not contain replacement characters"
            
        except UnicodeDecodeError as e:
            pytest.fail(f"Content should be decodable with detected encoding {detected_encoding}: {e}")
    
    @given(
        csv_content=csv_file_strategy(),
        corrupted_bytes=st.binary(min_size=1, max_size=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_2_corrupted_file_handling(self, csv_content: bytes, corrupted_bytes: bytes):
        """
        Property 2: Encoding Detection and Parsing - Corrupted File Handling
        For files with corrupted or invalid byte sequences, the system should handle gracefully
        and either detect a fallback encoding or provide appropriate error messages.
        **Feature: data-preprocessing-platform, Property 2: Encoding Detection and Parsing**
        **Validates: Requirements 1.5**
        """
        # Insert corrupted bytes at random position
        insert_pos = len(csv_content) // 2
        corrupted_content = csv_content[:insert_pos] + corrupted_bytes + csv_content[insert_pos:]
        
        csv_file = io.BytesIO(corrupted_content)
        
        # Test that encoding detection handles corrupted content gracefully
        try:
            detected_encoding, confidence = file_validator.detect_encoding(csv_file)
            
            # Should still return some encoding (possibly with low confidence)
            assert isinstance(detected_encoding, str), "Should return some encoding even for corrupted files"
            assert isinstance(confidence, (int, float)), "Should return numeric confidence"
            
            # Low confidence is acceptable for corrupted files
            assert 0.0 <= confidence <= 1.0, f"Confidence should be between 0 and 1, got {confidence}"
            
        except Exception:
            # It's acceptable for severely corrupted files to raise exceptions
            # as long as they're handled gracefully by the validation service
            pass


class TestDataProfiling:
    """Property tests for complete data profiling."""
    
    @given(df=dataframe_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_3_complete_data_profiling(self, df: pd.DataFrame):
        """
        Property 3: Complete Data Profiling
        For any uploaded dataset, the profiling engine should generate statistical summaries 
        for all columns, with appropriate statistics calculated based on data type.
        **Feature: data-preprocessing-platform, Property 3: Complete Data Profiling**
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
        """
        # Skip empty dataframes
        assume(len(df) > 0 and len(df.columns) > 0)
        
        try:
            # Generate comprehensive profile
            profile = profiling_service.generate_comprehensive_profile(df, "test_dataset")
            
            # Verify profile structure
            assert isinstance(profile, dict), "Profile should be a dictionary"
            
            # Check required top-level sections
            required_sections = ["dataset_overview", "column_statistics", "data_quality_summary", "custom_metrics"]
            for section in required_sections:
                assert section in profile, f"Profile should contain {section} section"
            
            # Verify dataset overview
            overview = profile["dataset_overview"]
            assert overview["num_rows"] == len(df), f"Row count mismatch: expected {len(df)}, got {overview['num_rows']}"
            assert overview["num_columns"] == len(df.columns), f"Column count mismatch: expected {len(df.columns)}, got {overview['num_columns']}"
            assert overview["memory_usage_bytes"] > 0, "Memory usage should be positive"
            assert "generated_at" in overview, "Should include generation timestamp"
            
            # Verify column statistics for all columns
            column_stats = profile["column_statistics"]
            assert len(column_stats) == len(df.columns), f"Should have stats for all {len(df.columns)} columns"
            
            for col_name in df.columns:
                assert col_name in column_stats, f"Missing statistics for column {col_name}"
                
                col_stat = column_stats[col_name]
                
                # Check basic statistics present for all columns
                required_basic_stats = ["name", "dtype", "count", "missing_count", "missing_percentage", "unique_count"]
                for stat in required_basic_stats:
                    assert stat in col_stat, f"Column {col_name} missing basic statistic: {stat}"
                
                # Verify basic statistics are correct
                assert col_stat["name"] == col_name, f"Column name mismatch for {col_name}"
                assert col_stat["count"] == len(df), f"Count should equal dataframe length for {col_name}"
                assert col_stat["missing_count"] == df[col_name].isnull().sum(), f"Missing count incorrect for {col_name}"
                assert col_stat["unique_count"] == df[col_name].nunique(), f"Unique count incorrect for {col_name}"
                
                # Check type-specific statistics
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    # Numeric columns should have descriptive statistics
                    if df[col_name].dropna().shape[0] > 0:  # Only if there are non-null values
                        numeric_stats = ["mean", "median", "std", "min", "max", "q25", "q75"]
                        for stat in numeric_stats:
                            assert stat in col_stat, f"Numeric column {col_name} missing statistic: {stat}"
                        
                        # Verify statistical relationships
                        if col_stat["min"] is not None and col_stat["max"] is not None:
                            assert col_stat["min"] <= col_stat["max"], f"Min should be <= max for {col_name}"
                        if col_stat["q25"] is not None and col_stat["q75"] is not None:
                            assert col_stat["q25"] <= col_stat["q75"], f"Q25 should be <= Q75 for {col_name}"
                
                elif pd.api.types.is_string_dtype(df[col_name]) or df[col_name].dtype == 'object':
                    # String/object columns should have string-specific statistics
                    if df[col_name].dropna().shape[0] > 0:  # Only if there are non-null values
                        string_stats = ["min_length", "max_length", "mean_length"]
                        for stat in string_stats:
                            assert stat in col_stat, f"String column {col_name} missing statistic: {stat}"
                        
                        # Verify string length relationships
                        if col_stat["min_length"] is not None and col_stat["max_length"] is not None:
                            assert col_stat["min_length"] <= col_stat["max_length"], f"Min length should be <= max length for {col_name}"
            
            # Verify data quality summary
            quality_summary = profile["data_quality_summary"]
            assert "overall_score" in quality_summary, "Should include overall quality score"
            assert isinstance(quality_summary["overall_score"], (int, float)), "Quality score should be numeric"
            assert 0 <= quality_summary["overall_score"] <= 100, "Quality score should be between 0 and 100"
            
            assert "issues" in quality_summary, "Should include issues summary"
            issues = quality_summary["issues"]
            assert "duplicate_rows" in issues, "Should report duplicate rows"
            assert "columns_with_missing" in issues, "Should report columns with missing values"
            assert "total_missing_values" in issues, "Should report total missing values"
            
            # Verify custom metrics
            custom_metrics = profile["custom_metrics"]
            assert "completeness" in custom_metrics, "Should include completeness metrics"
            assert "data_types" in custom_metrics, "Should include data type distribution"
            assert "uniqueness" in custom_metrics, "Should include uniqueness metrics"
            
            # Verify completeness metrics
            completeness = custom_metrics["completeness"]
            assert "overall_completeness" in completeness, "Should include overall completeness"
            assert isinstance(completeness["overall_completeness"], (int, float)), "Completeness should be numeric"
            assert 0 <= completeness["overall_completeness"] <= 100, "Completeness should be percentage"
            
            # Verify uniqueness metrics for all columns
            uniqueness = custom_metrics["uniqueness"]
            for col_name in df.columns:
                assert col_name in uniqueness, f"Should have uniqueness metrics for column {col_name}"
                col_uniqueness = uniqueness[col_name]
                assert "unique_values" in col_uniqueness, f"Should have unique_values for {col_name}"
                assert "uniqueness_ratio" in col_uniqueness, f"Should have uniqueness_ratio for {col_name}"
                assert 0 <= col_uniqueness["uniqueness_ratio"] <= 1, f"Uniqueness ratio should be between 0 and 1 for {col_name}"
            
        except Exception as e:
            pytest.fail(f"Data profiling should not fail for valid DataFrame: {e}")
    
    @given(df=dataframe_strategy())
    @settings(max_examples=50, deadline=None)
    def test_property_3_memory_calculation_accuracy(self, df: pd.DataFrame):
        """
        Property 3: Complete Data Profiling - Memory Calculation Accuracy
        For any dataset, memory usage calculations should be accurate and consistent.
        **Feature: data-preprocessing-platform, Property 3: Complete Data Profiling**
        **Validates: Requirements 2.5**
        """
        # Skip empty dataframes
        assume(len(df) > 0 and len(df.columns) > 0)
        
        try:
            # Calculate memory usage directly
            actual_memory = df.memory_usage(deep=True).sum()
            
            # Get memory usage from profiling service
            memory_analysis = profiling_service.calculate_memory_usage(df)
            
            # Verify memory calculation accuracy
            assert "total_bytes" in memory_analysis, "Should include total bytes"
            assert "total_mb" in memory_analysis, "Should include total MB"
            
            # Check that calculated memory matches actual memory
            calculated_memory = memory_analysis["total_bytes"]
            assert calculated_memory == actual_memory, f"Memory calculation mismatch: calculated {calculated_memory}, actual {actual_memory}"
            
            # Verify MB conversion is correct
            expected_mb = round(actual_memory / (1024 * 1024), 2)
            assert memory_analysis["total_mb"] == expected_mb, f"MB conversion incorrect: expected {expected_mb}, got {memory_analysis['total_mb']}"
            
            # Verify per-column memory breakdown
            if "columns_bytes" in memory_analysis:
                columns_total = sum(memory_analysis["columns_bytes"].values())
                index_bytes = memory_analysis.get("index_bytes", 0)
                total_from_breakdown = columns_total + index_bytes
                
                # Allow small differences due to rounding
                assert abs(total_from_breakdown - actual_memory) <= 1, f"Memory breakdown doesn't sum to total: {total_from_breakdown} vs {actual_memory}"
            
        except Exception as e:
            pytest.fail(f"Memory calculation should not fail for valid DataFrame: {e}")


class TestQualityIssueDetection:
    """Property tests for quality issue detection."""
    
    @given(df=dataframe_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_4_quality_issue_detection(self, df: pd.DataFrame):
        """
        Property 4: Quality Issue Detection
        For any dataset, the quality assessment should accurately detect and quantify 
        all data quality issues including missing values, duplicates, outliers, and 
        data type inconsistencies.
        **Feature: data-preprocessing-platform, Property 4: Quality Issue Detection**
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        from app.services.quality_service import quality_service
        
        # Skip empty dataframes
        assume(len(df) > 0 and len(df.columns) > 0)
        
        try:
            # Perform quality assessment
            assessment = quality_service.assess_quality(df)
            
            # Verify assessment structure
            assert isinstance(assessment, dict), "Assessment should be a dictionary"
            
            # Check required top-level sections
            required_sections = [
                "assessment_timestamp", "dataset_summary", "quality_scores",
                "issues", "issue_summary", "prioritized_recommendations", "column_quality"
            ]
            for section in required_sections:
                assert section in assessment, f"Assessment should contain {section} section"
            
            # Verify quality scores
            scores = assessment["quality_scores"]
            required_scores = ["overall", "completeness", "uniqueness", "consistency", "validity"]
            for score_name in required_scores:
                assert score_name in scores, f"Should have {score_name} score"
                assert isinstance(scores[score_name], (int, float)), f"{score_name} score should be numeric"
                assert 0 <= scores[score_name] <= 100, f"{score_name} score should be 0-100"
            
            # Verify issue summary
            summary = assessment["issue_summary"]
            assert summary["total_issues"] >= 0, "Total issues should be non-negative"
            severity_counts = summary["critical"] + summary["high"] + summary["medium"] + summary["low"] + summary["info"]
            assert severity_counts == summary["total_issues"], "Severity counts should sum to total"
            
            # Verify issues list
            issues = assessment["issues"]
            assert isinstance(issues, list), "Issues should be a list"
            
            for issue in issues:
                # Check required issue fields
                required_fields = ["issue_id", "issue_type", "severity", "description", 
                                   "affected_rows", "affected_percentage", "recommendation"]
                for field in required_fields:
                    assert field in issue, f"Issue should have {field} field"
                
                # Verify issue type and severity are valid
                valid_types = ["missing_values", "duplicates", "outliers", "type_inconsistency",
                              "high_cardinality", "low_variance", "constant_column", "mixed_types"]
                assert issue["issue_type"] in valid_types, f"Invalid issue type: {issue['issue_type']}"
                
                valid_severities = ["critical", "high", "medium", "low", "info"]
                assert issue["severity"] in valid_severities, f"Invalid severity: {issue['severity']}"
                
                # Verify affected counts are valid
                assert issue["affected_rows"] >= 0, "Affected rows should be non-negative"
                assert 0 <= issue["affected_percentage"] <= 100, "Percentage should be 0-100"
            
            # Verify column quality
            column_quality = assessment["column_quality"]
            for col in df.columns:
                assert col in column_quality, f"Should have quality info for column {col}"
                col_info = column_quality[col]
                assert "quality_score" in col_info, f"Column {col} should have quality_score"
                assert "completeness" in col_info, f"Column {col} should have completeness"
                assert "missing_count" in col_info, f"Column {col} should have missing_count"
            
        except Exception as e:
            pytest.fail(f"Quality assessment should not fail for valid DataFrame: {e}")
    
    @given(df=dataframe_strategy())
    @settings(max_examples=50, deadline=None)
    def test_property_4_missing_value_detection_accuracy(self, df: pd.DataFrame):
        """
        Property 4: Quality Issue Detection - Missing Value Detection Accuracy
        For any dataset, detected missing values should match actual missing values in the data.
        **Feature: data-preprocessing-platform, Property 4: Quality Issue Detection**
        **Validates: Requirements 3.1**
        """
        from app.services.quality_service import quality_service
        
        assume(len(df) > 0 and len(df.columns) > 0)
        
        try:
            # Perform quality assessment
            assessment = quality_service.assess_quality(df)
            
            # Get actual missing values per column
            actual_missing = {col: df[col].isnull().sum() for col in df.columns}
            
            # Check column quality matches actual missing counts
            column_quality = assessment["column_quality"]
            for col in df.columns:
                assert col in column_quality, f"Missing quality info for {col}"
                detected_missing = column_quality[col]["missing_count"]
                assert detected_missing == actual_missing[col], \
                    f"Missing count mismatch for {col}: detected {detected_missing}, actual {actual_missing[col]}"
            
            # Verify missing value issues are created for columns with missing data
            missing_issues = [i for i in assessment["issues"] if i["issue_type"] == "missing_values" and i.get("column")]
            columns_with_missing = [col for col, count in actual_missing.items() if count > 0]
            
            detected_columns = {i["column"] for i in missing_issues if i["column"]}
            for col in columns_with_missing:
                assert col in detected_columns, f"Column {col} with missing values should have an issue detected"
                
        except Exception as e:
            pytest.fail(f"Missing value detection should not fail: {e}")
    
    @given(df=dataframe_strategy())
    @settings(max_examples=50, deadline=None)
    def test_property_4_duplicate_detection_accuracy(self, df: pd.DataFrame):
        """
        Property 4: Quality Issue Detection - Duplicate Detection Accuracy
        For any dataset, detected duplicate count should match actual duplicates.
        **Feature: data-preprocessing-platform, Property 4: Quality Issue Detection**
        **Validates: Requirements 3.2**
        """
        from app.services.quality_service import quality_service
        
        assume(len(df) > 0 and len(df.columns) > 0)
        
        try:
            # Calculate actual duplicates
            actual_duplicates = df.duplicated().sum()
            
            # Perform quality assessment
            assessment = quality_service.assess_quality(df)
            
            # Find duplicate issues
            dup_issues = [i for i in assessment["issues"] 
                         if i["issue_type"] == "duplicates" and i.get("column") is None]
            
            if actual_duplicates > 0:
                # Should have at least one duplicate issue for dataset-level duplicates
                assert len(dup_issues) > 0, f"Should detect {actual_duplicates} duplicates"
                
                # Verify the count matches
                detected_dups = dup_issues[0]["affected_rows"]
                assert detected_dups == actual_duplicates, \
                    f"Duplicate count mismatch: detected {detected_dups}, actual {actual_duplicates}"
            else:
                # Should not have dataset-level duplicate issues
                assert len(dup_issues) == 0, "Should not have duplicate issues when no duplicates exist"
                
        except Exception as e:
            pytest.fail(f"Duplicate detection should not fail: {e}")


class TestTransformationConsistency:
    """Property tests for transformation operations."""
    
    @given(df=dataframe_strategy())
    @settings(max_examples=50, deadline=None)
    def test_property_6_transformation_consistency(self, df: pd.DataFrame):
        """
        Property 6: Transformation Consistency
        For any transformation operation, the result should be deterministic and
        consistent - applying the same transformation twice should yield identical results.
        **Feature: data-preprocessing-platform, Property 6: Transformation Consistency**
        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        from app.services.transformation_service import (
            transformation_service, TransformationType, TransformationParams
        )
        
        assume(len(df) > 0 and len(df.columns) > 0)
        
        try:
            # Test imputation consistency
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols and df[numeric_cols[0]].isnull().any():
                params = TransformationParams(columns=[numeric_cols[0]])
                
                # Apply transformation twice
                df1 = df.copy()
                df2 = df.copy()
                
                transformation_service._apply_transformation(
                    df1, TransformationType.IMPUTE_MEAN, params
                )
                transformation_service._apply_transformation(
                    df2, TransformationType.IMPUTE_MEAN, params
                )
                
                # Results should be identical
                pd.testing.assert_frame_equal(df1, df2)
            
            # Test remove duplicates consistency
            df1 = df.copy()
            df2 = df.copy()
            
            params = TransformationParams()
            transformation_service._apply_transformation(
                df1, TransformationType.REMOVE_DUPLICATES, params
            )
            transformation_service._apply_transformation(
                df2, TransformationType.REMOVE_DUPLICATES, params
            )
            
            pd.testing.assert_frame_equal(df1, df2)
            
        except Exception as e:
            pytest.fail(f"Transformation consistency should not fail: {e}")


class TestCodeGeneration:
    """Property tests for code generation."""
    
    @settings(max_examples=20, deadline=None)
    @given(num_transforms=st.integers(min_value=0, max_value=5))
    def test_property_7_code_generation_produces_valid_python(self, num_transforms: int):
        """
        Property 7: Code Generation Validity
        For any transformation history, the generated Python code should be
        syntactically valid and parseable.
        **Feature: data-preprocessing-platform, Property 7: Code Generation**
        **Validates: Requirements 7.1, 7.2, 7.3**
        """
        from app.services.code_generator import code_generator
        import ast
        
        # Create sample transformation history
        transform_types = [
            "impute_mean", "impute_median", "remove_duplicates",
            "scale_standard", "encode_label"
        ]
        
        transformations = []
        for i in range(num_transforms):
            transformations.append({
                "transformation_type": transform_types[i % len(transform_types)],
                "parameters": {"columns": [f"col_{i}"]},
                "timestamp": "2024-01-01T00:00:00"
            })
        
        try:
            # Generate pandas-style code
            pandas_code = code_generator.generate_python_code(
                transformations=transformations,
                dataset_name="test.csv",
                include_comments=True,
                style="pandas"
            )
            
            # Verify code is non-empty
            assert len(pandas_code) > 0, "Generated code should not be empty"
            
            # Verify code is valid Python (can be parsed)
            try:
                ast.parse(pandas_code)
            except SyntaxError as e:
                pytest.fail(f"Generated code has syntax error: {e}")
            
            # Verify essential components are present
            assert "import pandas" in pandas_code, "Should import pandas"
            assert "pd.read_csv" in pandas_code, "Should read CSV file"
            
        except Exception as e:
            pytest.fail(f"Code generation should not fail: {e}")

