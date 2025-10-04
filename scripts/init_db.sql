-- Initialize database for Medical Imaging AI API

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS medical_imaging_api;

-- Use the database
\c medical_imaging_api;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better performance
-- These will be created by SQLAlchemy, but we can add custom ones here

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE medical_imaging_api TO postgres;
