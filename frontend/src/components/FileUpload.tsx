/**
 * FileUpload Component
 * Drag-and-drop file upload with progress indicator
 */
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import { datasetApi, Dataset } from '@/services/api';

interface FileUploadProps {
    onUploadSuccess?: (dataset: Dataset) => void;
    onUploadError?: (error: Error) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({
    onUploadSuccess,
    onUploadError,
}) => {
    const [uploadProgress, setUploadProgress] = useState<number>(0);

    const uploadMutation = useMutation({
        mutationFn: (file: File) => datasetApi.upload(file, setUploadProgress),
        onSuccess: (data) => {
            setUploadProgress(100);
            onUploadSuccess?.(data);
        },
        onError: (error: Error) => {
            setUploadProgress(0);
            onUploadError?.(error);
        },
    });

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            setUploadProgress(0);
            uploadMutation.mutate(acceptedFiles[0]);
        }
    }, [uploadMutation]);

    const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
        onDrop,
        accept: {
            'text/csv': ['.csv'],
        },
        maxFiles: 1,
        maxSize: 1024 * 1024 * 1024, // 1GB
    });

    return (
        <div className="w-full">
            <div
                {...getRootProps()}
                className={`
          relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
          transition-all duration-300 ease-in-out
          ${isDragActive && !isDragReject ? 'border-primary bg-primary/5 scale-[1.02]' : ''}
          ${isDragReject ? 'border-red-500 bg-red-500/5' : ''}
          ${!isDragActive && !isDragReject ? 'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50' : ''}
          ${uploadMutation.isPending ? 'pointer-events-none opacity-75' : ''}
        `}
            >
                <input {...getInputProps()} />

                <div className="flex flex-col items-center gap-4">
                    {/* Upload Icon */}
                    <div className={`
            w-16 h-16 rounded-full flex items-center justify-center
            ${isDragActive ? 'bg-primary/10' : 'bg-muted'}
            transition-colors duration-300
          `}>
                        <svg
                            className={`w-8 h-8 ${isDragActive ? 'text-primary' : 'text-muted-foreground'}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                            />
                        </svg>
                    </div>

                    {/* Text */}
                    <div>
                        <p className="text-lg font-semibold text-foreground">
                            {isDragActive ? 'Drop your file here' : 'Drag & drop your CSV file'}
                        </p>
                        <p className="text-sm text-muted-foreground mt-1">
                            or click to browse • Max 1GB
                        </p>
                    </div>

                    {/* File type badge */}
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-muted text-muted-foreground text-sm">
                        <span>📄</span>
                        <span>CSV files only</span>
                    </div>
                </div>

                {/* Progress bar */}
                {uploadMutation.isPending && (
                    <div className="absolute bottom-0 left-0 right-0 p-4">
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div
                                className="h-full bg-primary transition-all duration-300 ease-out"
                                style={{ width: `${uploadProgress}%` }}
                            />
                        </div>
                        <p className="text-sm text-muted-foreground text-center mt-2">
                            Uploading... {uploadProgress}%
                        </p>
                    </div>
                )}
            </div>

            {/* Error message */}
            {uploadMutation.isError && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-600 text-sm">
                        {uploadMutation.error?.message || 'Upload failed. Please try again.'}
                    </p>
                </div>
            )}

            {/* Success message */}
            {uploadMutation.isSuccess && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-green-600 text-sm">
                        ✓ File uploaded successfully! Redirecting to analysis...
                    </p>
                </div>
            )}
        </div>
    );
};

export default FileUpload;
