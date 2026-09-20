import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile


# -------------------------------------------------------
# Upload Directory
# -------------------------------------------------------

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------------------------------------
# Generate Unique Filename
# -------------------------------------------------------

def generate_filename(
    original_filename: str,
) -> str:
    """
    Example:

    report.pdf

    becomes

    3fd8322a9_report.pdf
    """

    unique_id = uuid.uuid4().hex

    return f"{unique_id}_{original_filename}"


# -------------------------------------------------------
# Save Uploaded File
# -------------------------------------------------------

def save_file(
    file: UploadFile,
) -> tuple[str, str]:
    """
    Saves uploaded file.

    Returns

    (
        stored_filename,
        stored_path
    )
    """

    filename = generate_filename(
        file.filename
    )

    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    return (
        filename,
        str(file_path),
    )


# -------------------------------------------------------
# Delete File
# -------------------------------------------------------

def delete_file(
    file_path: str,
) -> bool:

    path = Path(file_path)

    if path.exists():
        path.unlink()
        return True

    return False


# -------------------------------------------------------
# Check File Exists
# -------------------------------------------------------

def file_exists(
    file_path: str,
) -> bool:

    return Path(file_path).exists()