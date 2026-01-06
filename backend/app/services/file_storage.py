"""
File storage service for handling file uploads and storage operations.
"""
import os
import uuid
from typing import BinaryIO, Optional
from minio import Minio
from minio.error import S3Error
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileStorageService:
    """Service for handling file storage operations with MinIO/S3."""
    
    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Ensure the storage bucket exists."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    def store_file(self, file_content: BinaryIO, filename: str, content_type: str = "application/octet-stream") -> str:
        """
        Store a file in MinIO and return the file path.
        
        Args:
            file_content: File content as binary stream
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            str: Unique file path in storage
            
        Raises:
            S3Error: If storage operation fails
        """
        try:
            # Generate unique file path
            file_extension = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = f"{settings.UPLOAD_DIR}/{unique_filename}"
            
            # Store file in MinIO
            file_content.seek(0)  # Reset file pointer
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=file_path,
                data=file_content,
                length=-1,  # Unknown length, MinIO will determine
                part_size=10*1024*1024,  # 10MB parts
                content_type=content_type
            )
            
            logger.info(f"Successfully stored file: {file_path}")
            return file_path
            
        except S3Error as e:
            logger.error(f"Error storing file {filename}: {e}")
            raise
    
    def get_file(self, file_path: str) -> BinaryIO:
        """
        Retrieve a file from storage.
        
        Args:
            file_path: Path to the file in storage
            
        Returns:
            BinaryIO: File content as binary stream
            
        Raises:
            S3Error: If file retrieval fails
        """
        try:
            response = self.client.get_object(self.bucket_name, file_path)
            return response
        except S3Error as e:
            logger.error(f"Error retrieving file {file_path}: {e}")
            raise
    
    def delete_file(self, file_path: str) -> None:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path to the file in storage
            
        Raises:
            S3Error: If file deletion fails
        """
        try:
            self.client.remove_object(self.bucket_name, file_path)
            logger.info(f"Successfully deleted file: {file_path}")
        except S3Error as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path to the file in storage
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.client.stat_object(self.bucket_name, file_path)
            return True
        except S3Error:
            return False


# Global instance
file_storage = FileStorageService()