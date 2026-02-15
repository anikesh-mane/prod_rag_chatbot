"""Chat endpoint for RAG-based question answering."""

import time
from typing import AsyncGenerator
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.dependencies import (
    ConversationManagerDep,
    GeneratorDep,
    OptionalUserDep,
    RetrieverDep,
    SettingsDep,
)
from schemas import (
    ChatRequest,
    ChatResponse,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    SourceDocument,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def chat(
    request: Request,
    body: ChatRequest,
    retriever: RetrieverDep,
    generator: GeneratorDep,
    settings: SettingsDep,
    current_user: OptionalUserDep,
    conversation_manager: ConversationManagerDep,
) -> ChatResponse:
    """Process a chat query using RAG pipeline.

    1. Retrieve relevant documents from vector store
    2. Generate response using LLM with retrieved context
    3. Return response with sources
    """
    start_time = time.perf_counter()
    query_id = uuid4()
    user_id = current_user.user_id if current_user else None

    logger.info(
        "Chat request received",
        query_id=str(query_id),
        query_length=len(body.query),
        user_id=user_id or "anonymous",
    )

    try:
        # Get or create conversation
        conversation_id = await conversation_manager.get_or_create_conversation(
            conversation_id=body.conversation_id,
            user_id=user_id,
        )

        # Load conversation history (Redis first, PostgreSQL fallback)
        conv_context = await conversation_manager.get_context(conversation_id)
        conversation_history = conv_context.messages if conv_context.message_count > 0 else None

        # Retrieve relevant documents
        results = await retriever.retrieve(
            query=body.query,
            top_k=body.top_k,
        )

        if not results:
            logger.warning("No relevant documents found", query_id=str(query_id))
            no_context_answer = "I don't have enough information in my knowledge base to answer this question."

            # Save turn even for no-context responses
            await conversation_manager.add_turn(
                conversation_id=conversation_id,
                user_message=body.query,
                assistant_message=no_context_answer,
                query_id=query_id,
            )

            return ChatResponse(
                query_id=query_id,
                answer=no_context_answer,
                language=body.language or "en",
                sources=[],
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model="none",
                conversation_id=str(conversation_id),
            )

        # Generate response with conversation history
        answer, llm_response = await generator.generate(
            query=body.query,
            context=results,
            conversation_history=conversation_history,
        )

        # Build source documents for response
        sources: list[SourceDocument] = []
        if body.include_sources:
            for result in results:
                sources.append(
                    SourceDocument(
                        document_id=result.document_id,
                        content=result.content[:500],  # Truncate for response
                        score=result.score,
                        metadata=result.metadata,
                    )
                )

        # Save turn to memory (PostgreSQL + Redis)
        source_refs = [
            {"document_id": src.document_id, "score": src.score}
            for src in sources
        ] if sources else None

        await conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_message=body.query,
            assistant_message=answer,
            query_id=query_id,
            sources=source_refs,
            tokens_used=llm_response.total_tokens,
            model=llm_response.model,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Chat response generated",
            query_id=str(query_id),
            conversation_id=str(conversation_id),
            sources_count=len(sources),
            tokens=llm_response.total_tokens,
            latency_ms=round(latency_ms, 2),
        )

        return ChatResponse(
            query_id=query_id,
            answer=answer,
            language=body.language or "en",
            sources=sources,
            tokens_used=llm_response.total_tokens,
            latency_ms=latency_ms,
            model=llm_response.model,
            conversation_id=str(conversation_id),
        )

    except Exception as e:
        logger.error(
            "Chat request failed",
            query_id=str(query_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to process chat request",
                request_id=getattr(request.state, "request_id", query_id),
                details={"error": str(e)} if settings.is_development else None,
            ).model_dump(),
        )


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    retriever: RetrieverDep,
    generator: GeneratorDep,
    settings: SettingsDep,
    current_user: OptionalUserDep,
) -> StreamingResponse:
    """Stream a chat response using Server-Sent Events.

    Returns response chunks as they are generated by the LLM.
    """
    query_id = uuid4()

    logger.info(
        "Streaming chat request received",
        query_id=str(query_id),
        query_length=len(body.query),
    )

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Retrieve relevant documents
            results = await retriever.retrieve(
                query=body.query,
                top_k=body.top_k,
            )

            if not results:
                yield f"data: {{'error': 'No relevant documents found'}}\n\n"
                return

            # Stream response
            async for chunk in generator.generate_stream(
                query=body.query,
                context=results,
            ):
                # Format as SSE
                yield f"data: {chunk}\n\n"

            # Send done signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(
                "Streaming chat failed",
                query_id=str(query_id),
                error=str(e),
            )
            yield f"data: {{'error': 'Stream failed: {str(e)}'}}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Query-ID": str(query_id),
        },
    )
