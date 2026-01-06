-- DataPrep AI Platform Database Initialization
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE NOT NULL
);

-- Create indexes for users table
CREATE INDEX IF NOT EXISTS ix_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

-- Create datasets table
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    num_rows INTEGER,
    num_columns INTEGER,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'uploaded' NOT NULL,
    column_metadata JSONB
);

-- Create indexes for datasets table
CREATE INDEX IF NOT EXISTS ix_datasets_dataset_id ON datasets(dataset_id);
CREATE INDEX IF NOT EXISTS ix_datasets_user_id ON datasets(user_id);

-- Create transformations table
CREATE TABLE IF NOT EXISTS transformations (
    transformation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    operation_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sequence_order INTEGER NOT NULL
);

-- Create indexes for transformations table
CREATE INDEX IF NOT EXISTS ix_transformations_transformation_id ON transformations(transformation_id);
CREATE INDEX IF NOT EXISTS ix_transformations_dataset_id ON transformations(dataset_id);

-- Create analysis_cache table
CREATE TABLE IF NOT EXISTS analysis_cache (
    cache_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    analysis_type VARCHAR(100) NOT NULL,
    results_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for analysis_cache table
CREATE INDEX IF NOT EXISTS ix_analysis_cache_cache_id ON analysis_cache(cache_id);
CREATE INDEX IF NOT EXISTS ix_analysis_cache_dataset_id ON analysis_cache(dataset_id);

-- Insert initial test data for development
INSERT INTO users (email, is_active) VALUES 
    ('test@dataprep.ai', true),
    ('admin@dataprep.ai', true)
ON CONFLICT (email) DO NOTHING;