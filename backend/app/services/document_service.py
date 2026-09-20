from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.user import User
from app.rag.parser import parse_document

from app.utils.file_storage import save_file, delete_file
from app.utils.file_validator import validate_upload
from app.utils.storage_limits import validate_user_storage_quota


class DocumentService:
    """
    Handles all document-related operations.

    Features
    --------
    • Upload document
    • Parse uploaded document
    • Upload multiple documents
    • List user documents
    • Get document
    • Delete document
    • Download document
    """

    def __init__(self, db: Session):
        self.db = db

    # =====================================================
    # Upload Single File
    # =====================================================

    def upload_document(
        self,
        file: UploadFile,
        owner: User,
    ) -> Document:

        # =================================================
        # 1. Validate file
        # =================================================

        validate_upload(file)

        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        validate_user_storage_quota(
            self.db,
            owner.id,
            size,
        )

        # =================================================
        # 2. Save file locally
        # =================================================

        filename, file_path = save_file(file)

        try:

            # =============================================
            # 3. Parse uploaded document
            #
            # Phase 5 supported formats:
            # PDF, DOCX, TXT, CSV, XLSX
            # =============================================

            parsed_document = parse_document(
                Path(file_path)
            )

            # =============================================
            # 4. File size was validated before persistence
            # =============================================

            # =============================================
            # 5. Create PostgreSQL document record
            # =============================================

            document = Document(
                filename=filename,
                original_filename=file.filename,
                file_path=file_path,
                file_type=Path(file.filename).suffix.lower(),
                file_size=size,
                owner_id=owner.id,
            )

            # =============================================
            # 6. Save document to PostgreSQL
            # =============================================

            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)

            # =============================================
            # 7. Log ingestion result
            #
            # These values are used only for verification
            # at this stage.
            #
            # Chunk persistence, embeddings, ChromaDB,
            # multimodal processing and RAG will be added
            # in later integration steps.
            # =============================================

            print(
                f"[INGESTION] SUCCESS: "
                f"{file.filename}"
            )

            print(
                f"[INGESTION] "
                f"Text blocks: "
                f"{len(parsed_document.text_blocks)}"
            )

            print(
                f"[INGESTION] "
                f"Tables: "
                f"{len(parsed_document.tables)}"
            )

            print(
                f"[INGESTION] "
                f"Images: "
                f"{len(parsed_document.images)}"
            )

            print(
                f"[INGESTION] "
                f"Charts: "
                f"{len(parsed_document.charts)}"
            )

            return document

        except Exception:

            # =============================================
            # 8. Roll back database transaction
            # =============================================

            self.db.rollback()

            # =============================================
            # 9. Delete saved file if parsing/database
            #    processing fails
            # =============================================

            try:
                delete_file(file_path)
            except Exception:
                pass

            # =============================================
            # 10. Re-raise original exception
            # =============================================

            raise

    # =====================================================
    # Upload Multiple Files
    # =====================================================

    def upload_multiple_documents(
        self,
        files: list[UploadFile],
        owner: User,
    ) -> list[Document]:

        documents = []

        for file in files:

            document = self.upload_document(
                file=file,
                owner=owner,
            )

            documents.append(document)

        return documents

    # =====================================================
    # Get All User Documents
    # =====================================================

    def get_documents(
        self,
        owner: User,
    ) -> list[Document]:

        return (
            self.db.query(Document)
            .filter(
                Document.owner_id == owner.id
            )
            .order_by(
                Document.created_at.desc()
            )
            .all()
        )

    # =====================================================
    # Get Document By ID
    # =====================================================

    def get_document(
        self,
        document_id: UUID,
        owner: User,
    ) -> Document:

        document = (
            self.db.query(Document)
            .filter(
                Document.id == document_id,
                Document.owner_id == owner.id,
            )
            .first()
        )

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        return document

    # =====================================================
    # Delete Document
    # =====================================================

    def delete_document(
        self,
        document_id: UUID,
        owner: User,
    ) -> dict:

        document = self.get_document(
            document_id,
            owner,
        )

        self.db.delete(document)
        self.db.commit()

        return {
            "success": True,
            "message": "Document deleted successfully.",
        }

    # =====================================================
    # Download Document
    # =====================================================

    def download_document(
        self,
        document_id: UUID,
        owner: User,
    ) -> Document:

        return self.get_document(
            document_id,
            owner,
        )