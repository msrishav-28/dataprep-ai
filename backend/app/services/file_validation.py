"""
File validation service for validating uploaded files.
"""
import os
import chardet
from typing import BinaryIO, Tuple, Optional
from fastapi import HTTPException, status
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileValidationService:
    """Service for validating uploaded files."""
    
    @staticmethod
    def validate_file_size(file_size: int) -> None:
        """
        Validate file size against maximum allowed size.
        
        Args:
            file_size: Size of the file in bytes
            
        Raises:
            HTTPException: If file size exceeds limit
        """
        if file_size > settings.MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_size_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb:.0f}MB)"
            )
    
    @staticmethod
    def validate_file_format(filename: str) -> None:
        """
        Validate file format against allowed extensions.
        
        Args:
            filename: Name of the uploaded file
            
        Raises:
            HTTPException: If file format is not allowed
        """
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
            allowed_formats = ", ".join(settings.ALLOWED_FILE_EXTENSIONS)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File format '{file_extension}' not supported. Allowed formats: {allowed_formats}"
            )
    
    @staticmethod
    def detect_encoding(file_content: BinaryIO, sample_size: int = 10000) -> Tuple[str, float]:
        """
        Detect the character encoding of a file.
        
        Args:
            file_content: File content as binary stream
            sample_size: Number of bytes to sample for detection
            
        Returns:
            Tuple[str, float]: Detected encoding and confidence score
            
        Raises:
            HTTPException: If encoding detection fails
        """
        try:
            # Read sample for encoding detection
            file_content.seek(0)
            sample = file_content.read(sample_size)
            file_content.seek(0)  # Reset file pointer
            
            # Detect encoding
            detection_result = chardet.detect(sample)
            
            if detection_result['encoding'] is None:
                logger.warning("Could not detect file encoding, defaulting to UTF-8")
                return 'utf-8', 0.0
            
            encoding = detection_result['encoding'].lower()
            confidence = detection_result['confidence']
            
            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Validate that we can decode with detected encoding
            try:
                sample.decode(encoding)
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode with detected encoding {encoding}, falling back to UTF-8")
                encoding = 'utf-8'
                confidence = 0.0
            
            return encoding, confidence
            
        except Exception as e:
            logger.error(f"Error detecting file encoding: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to detect file encoding. Please ensure the file is a valid text file."
            )
    
    @staticmethod
    def validate_csv_content(file_content: BinaryIO, encoding: str) -> None:
        """
        Perform basic validation of CSV content.
        
        Args:
            file_content: File content as binary stream
            encoding: Character encoding to use for reading
            
        Raises:
            HTTPException: If CSV content is invalid
        """
        try:
            file_content.seek(0)
            
            # Read first few lines to validate CSV structure
            lines_to_check = 5
            lines_read = 0
            
            for line in file_content:
                try:
                    decoded_line = line.decode(encoding).strip()
                    if decoded_line:  # Skip empty lines
                        lines_read += 1
                        if lines_read >= lines_to_check:
                            break
                except UnicodeDecodeError as e:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"File contains invalid characters for encoding {encoding}: {str(e)}"
                    )
            
            if lines_read == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="File appears to be empty or contains no valid data"
                )
            
            file_content.seek(0)  # Reset file pointer
            logger.info(f"CSV content validation passed ({lines_read} lines checked)")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating CSV content: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to validate CSV content. Please ensure the file is a valid CSV file."
            )


# Global instance
file_validator = FileValidationService()