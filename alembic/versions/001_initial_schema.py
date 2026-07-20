"""initial schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-07-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pdfs",
        sa.Column("pdf_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("collection_name", sa.String(), nullable=False),
        sa.Column("upload_timestamp", sa.DateTime(), nullable=False),
        sa.Column("doc_count", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("is_sample", sa.Boolean(), nullable=True),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("pdf_id"),
        sa.UniqueConstraint("collection_name"),
    )
    op.create_table(
        "chat_sessions",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_active", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_table(
        "messages",
        sa.Column("message_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("message_id"),
    )


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("chat_sessions")
    op.drop_table("pdfs")
