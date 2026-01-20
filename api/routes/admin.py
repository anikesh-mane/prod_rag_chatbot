"""Admin endpoints for document management."""

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from api.dependencies import (
    AdminUserDep,
    IngestionDep,
    SettingsDep,
    VectorStoreDep,
)
from ingestion.embedders import get_embedder
from schemas import (
    ChunkSummary,
    DeleteDocumentResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    ReindexResponse,
    VectorStoreStatsResponse,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get(
    "/admin/documents",
    response_model=DocumentListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - Admin role required"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_documents(
    request: Request,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: AdminUserDep,
    limit: int = 50,
    offset: int = 0,
) -> DocumentListResponse:
    """List all documents in the vector store.

    Requires admin role. Returns paginated list of documents with chunk counts.
    """
    # Validate pagination params
    if limit < 1 or limit > 500:
        limit = 50
    if offset < 0:
        offset = 0

    logger.info(
        "Admin listing documents",
        user_id=current_user.user_id,
        limit=limit,
        offset=offset,
    )

    try:
        docs, total = await vector_store.list_documents(limit=limit, offset=offset)

        documents = [
            DocumentSummary(
                document_id=doc["document_id"],
                chunk_count=doc["chunk_count"],
                metadata=doc.get("metadata", {}),
            )
            for doc in docs
        ]

        return DocumentListResponse(
            documents=documents,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(
            "Failed to list documents",
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to list documents",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )


@router.get(
    "/admin/documents/{document_id}",
    response_model=DocumentDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - Admin role required"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document(
    request: Request,
    document_id: str,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: AdminUserDep,
) -> DocumentDetailResponse:
    """Get detailed information about a specific document.

    Requires admin role. Returns document metadata and all chunks.
    """
    logger.info(
        "Admin retrieving document",
        document_id=document_id,
        user_id=current_user.user_id,
    )

    try:
        chunks = await vector_store.get_document_chunks(document_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Document not found: {document_id}",
                    request_id=getattr(request.state, "request_id", None),
                ).model_dump(),
            )

        # Build chunk summaries with content preview
        chunk_summaries = [
            ChunkSummary(
                chunk_id=chunk["chunk_id"],
                content_preview=chunk["content"][:200] if chunk.get("content") else "",
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks
        ]

        # Get metadata from first chunk
        metadata = chunks[0].get("metadata", {}) if chunks else {}

        return DocumentDetailResponse(
            document_id=document_id,
            chunk_count=len(chunks),
            chunks=chunk_summaries,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get document",
            document_id=document_id,
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to retrieve document",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )


@router.delete(
    "/admin/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - Admin role required"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_document(
    request: Request,
    document_id: str,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: AdminUserDep,
) -> DeleteDocumentResponse:
    """Delete a document and all its chunks from the vector store.

    Requires admin role. This action is irreversible.
    """
    logger.info(
        "Admin deleting document",
        document_id=document_id,
        user_id=current_user.user_id,
    )

    try:
        # Check if document exists first
        doc_info = await vector_store.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Document not found: {document_id}",
                    request_id=getattr(request.state, "request_id", None),
                ).model_dump(),
            )

        # Delete the document
        deleted_count = await vector_store.delete_by_document(document_id)

        logger.info(
            "Document deleted",
            document_id=document_id,
            chunks_deleted=deleted_count,
            user_id=current_user.user_id,
        )

        return DeleteDocumentResponse(
            document_id=document_id,
            chunks_deleted=deleted_count,
            status="deleted",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete document",
            document_id=document_id,
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to delete document",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )


@router.post(
    "/admin/documents/{document_id}/reindex",
    response_model=ReindexResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - Admin role required"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reindex_document(
    request: Request,
    document_id: str,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: AdminUserDep,
) -> ReindexResponse:
    """Re-generate embeddings for a document.

    Requires admin role. Fetches existing chunks, generates new embeddings
    with the current embedder, and updates the vector store.
    """
    logger.info(
        "Admin re-indexing document",
        document_id=document_id,
        user_id=current_user.user_id,
    )

    try:
        # Get existing chunks
        chunks = await vector_store.get_document_chunks(document_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Document not found: {document_id}",
                    request_id=getattr(request.state, "request_id", None),
                ).model_dump(),
            )

        # Extract chunk data
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        metadata_list = [chunk.get("metadata", {}) for chunk in chunks]
        document_ids = [document_id] * len(chunks)

        # Delete existing chunks
        await vector_store.delete_by_document(document_id)

        # Generate new embeddings
        embedder = get_embedder()
        new_embeddings = await embedder.embed_batch(contents)

        # Re-insert with new embeddings
        await vector_store.insert(
            chunk_ids=chunk_ids,
            document_ids=document_ids,
            contents=contents,
            embeddings=new_embeddings,
            metadata=metadata_list,
        )

        logger.info(
            "Document re-indexed",
            document_id=document_id,
            chunks_reindexed=len(chunks),
            user_id=current_user.user_id,
        )

        return ReindexResponse(
            document_id=document_id,
            chunks_reindexed=len(chunks),
            status="reindexed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to re-index document",
            document_id=document_id,
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to re-index document",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )


@router.get(
    "/admin/stats",
    response_model=VectorStoreStatsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - Admin role required"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_stats(
    request: Request,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: AdminUserDep,
) -> VectorStoreStatsResponse:
    """Get vector store statistics.

    Requires admin role. Returns collection name, total chunks, and index info.
    """
    logger.info(
        "Admin retrieving stats",
        user_id=current_user.user_id,
    )

    try:
        stats = await vector_store.get_stats()

        return VectorStoreStatsResponse(
            collection_name=stats["collection_name"],
            total_chunks=stats["num_entities"],
            dimension=stats["dimension"],
            index_type=stats["index_type"],
            metric_type=stats["metric_type"],
        )

    except Exception as e:
        logger.error(
            "Failed to get stats",
            error=str(e),
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to retrieve statistics",
                request_id=getattr(request.state, "request_id", None),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )
