"""Document ingestion endpoints."""

import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from api.dependencies import (
    CurrentUserDep,
    IngestionDep,
    SettingsDep,
    VectorStoreDep,
)
from schemas import (
    DocumentIngestRequest,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    FileIngestResponse,
    IngestResponse,
)

router = APIRouter()
logger = structlog.get_logger(__name__)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def ingest_text(
    request: Request,
    body: DocumentIngestRequest,
    pipeline: IngestionDep,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> IngestResponse:
    """Ingest raw text content into the vector store.

    Requires authentication. The text will be chunked, embedded, and stored
    in the vector database for retrieval.
    """
    logger.info(
        "Text ingestion request",
        source=body.source,
        content_length=len(body.content),
        user_id=current_user.user_id,
    )

    try:
        # Process document through ingestion pipeline
        document, chunks, embeddings = await pipeline.ingest_text(
            content=body.content,
            source=body.source,
            metadata=body.metadata,
        )

        # Store in vector database
        chunk_ids = [str(chunk.chunk_id) for chunk in chunks]
        document_ids = [str(document.document_id)] * len(chunks)
        contents = [chunk.content for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]

        await vector_store.insert(
            chunk_ids=chunk_ids,
            document_ids=document_ids,
            contents=contents,
            embeddings=embeddings,
            metadata=metadata_list,
        )

        logger.info(
            "Text ingestion complete",
            document_id=str(document.document_id),
            chunks_created=len(chunks),
            user_id=current_user.user_id,
        )

        return IngestResponse(
            document_id=document.document_id,
            chunks_created=len(chunks),
            status="ingested",
        )

    except Exception as e:
        logger.error(
            "Text ingestion failed",
            source=body.source,
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to ingest document",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )


@router.post(
    "/ingest/file",
    response_model=FileIngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or size"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def ingest_file(
    request: Request,
    file: UploadFile = File(..., description="Document file to ingest"),
    pipeline: IngestionDep = None,
    vector_store: VectorStoreDep = None,
    settings: SettingsDep = None,
    current_user: CurrentUserDep = None,
) -> FileIngestResponse:
    """Upload and ingest a document file.

    Requires authentication. Supported formats: PDF, DOCX, TXT, MD.
    Maximum file size: 10MB.
    """
    # Validate file extension
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    logger.info(
        "File ingestion request",
        filename=filename,
        file_type=file_ext,
        user_id=current_user.user_id,
    )

    # Read file content and check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    temp_path = None
    try:
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Process through ingestion pipeline
        document, chunks, embeddings = await pipeline.ingest_file(temp_path)

        # Store in vector database
        chunk_ids = [str(chunk.chunk_id) for chunk in chunks]
        document_ids = [str(document.document_id)] * len(chunks)
        contents = [chunk.content for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]

        await vector_store.insert(
            chunk_ids=chunk_ids,
            document_ids=document_ids,
            contents=contents,
            embeddings=embeddings,
            metadata=metadata_list,
        )

        logger.info(
            "File ingestion complete",
            filename=filename,
            document_id=str(document.document_id),
            chunks_created=len(chunks),
            user_id=current_user.user_id,
        )

        return FileIngestResponse(
            document_id=document.document_id,
            filename=filename,
            file_type=file_ext.lstrip("."),
            chunks_created=len(chunks),
            status="ingested",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "File ingestion failed",
            filename=filename,
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to ingest file",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )
    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()
