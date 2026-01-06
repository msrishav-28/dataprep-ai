"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-12-25 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_user_id'), 'users', ['user_id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)

    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('num_rows', sa.Integer(), nullable=True),
        sa.Column('num_columns', sa.Integer(), nullable=True),
        sa.Column('upload_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, default='uploaded'),
        sa.Column('column_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('dataset_id')
    )
    op.create_index(op.f('ix_datasets_dataset_id'), 'datasets', ['dataset_id'], unique=False)
    op.create_index(op.f('ix_datasets_user_id'), 'datasets', ['user_id'], unique=False)

    # Create transformations table
    op.create_table(
        'transformations',
        sa.Column('transformation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('operation_type', sa.String(length=100), nullable=False),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('sequence_order', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.dataset_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('transformation_id')
    )
    op.create_index(op.f('ix_transformations_transformation_id'), 'transformations', ['transformation_id'], unique=False)
    op.create_index(op.f('ix_transformations_dataset_id'), 'transformations', ['dataset_id'], unique=False)

    # Create analysis_cache table
    op.create_table(
        'analysis_cache',
        sa.Column('cache_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('analysis_type', sa.String(length=100), nullable=False),
        sa.Column('results_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.dataset_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('cache_id')
    )
    op.create_index(op.f('ix_analysis_cache_cache_id'), 'analysis_cache', ['cache_id'], unique=False)
    op.create_index(op.f('ix_analysis_cache_dataset_id'), 'analysis_cache', ['dataset_id'], unique=False)


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('analysis_cache')
    op.drop_table('transformations')
    op.drop_table('datasets')
    op.drop_table('users')