from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "xxxxxxxx"
down_revision = "your_previous_revision"


def upgrade():

    op.create_table(

        "documents",

        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),

        sa.Column(
            "filename",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "original_filename",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "file_path",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "file_type",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "file_size",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),

        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),

        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():

    op.drop_table("documents")