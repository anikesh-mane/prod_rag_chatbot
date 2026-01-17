"""LLM generation module combining templates and clients."""

from typing import AsyncGenerator
from uuid import UUID, uuid4

import structlog

from configs import get_settings
from llm.clients import get_llm_client, BaseLLMClient
from llm.templates import get_template
from schemas import LLMRequest, LLMResponse, QueryContext, RetrievalResult

logger = structlog.get_logger(__name__)


class RAGGenerator:
    """Generator for RAG-based responses."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        template_name: str = "rag_qa",
    ):
        """Initialize RAG generator.

        Args:
            llm_client: LLM client (created if not provided).
            template_name: Name of prompt template to use.
        """
        settings = get_settings()
        self._client = llm_client or get_llm_client()
        self._template = get_template(template_name)
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens

    async def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, LLMResponse]:
        """Generate a RAG response.

        Args:
            query: User query.
            context: Retrieved context documents.
            conversation_history: Optional conversation history.

        Returns:
            Tuple of (generated_answer, llm_response).
        """
        # Format context for prompt
        formatted_context = [
            {
                "source": result.metadata.get("file_name", result.document_id),
                "content": result.content,
            }
            for result in context
        ]

        # Build prompt
        if conversation_history and self._template.name == "rag_qa_with_history":
            prompt = self._template.format(
                context=formatted_context,
                question=query,
                history=conversation_history,
            )
        else:
            prompt = self._template.format(
                context=formatted_context,
                question=query,
            )

        logger.debug(
            "Generated prompt",
            template=self._template.name,
            context_docs=len(context),
            prompt_length=len(prompt),
        )

        # Generate response
        request = LLMRequest(
            prompt=prompt,
            model=self._client.model_name.split(" ")[0],  # Handle fallback naming
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        response = await self._client.generate(request)

        logger.info(
            "RAG generation complete",
            tokens=response.total_tokens,
            latency_ms=round(response.latency_ms, 2),
        )

        return response.content, response

    async def generate_stream(
        self,
        query: str,
        context: list[RetrievalResult],
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming RAG response.

        Args:
            query: User query.
            context: Retrieved context documents.

        Yields:
            Response content chunks.
        """
        # Format context for prompt
        formatted_context = [
            {
                "source": result.metadata.get("file_name", result.document_id),
                "content": result.content,
            }
            for result in context
        ]

        prompt = self._template.format(
            context=formatted_context,
            question=query,
        )

        request = LLMRequest(
            prompt=prompt,
            model=self._client.model_name.split(" ")[0],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        async for chunk in self._client.generate_stream(request):
            yield chunk

    async def create_query_context(
        self,
        query: str,
        context: list[RetrievalResult],
        response: str,
        llm_response: LLMResponse,
    ) -> QueryContext:
        """Create a query context for logging/feedback.

        Args:
            query: Original user query.
            context: Retrieved documents.
            response: Generated response.
            llm_response: LLM response metadata.

        Returns:
            QueryContext for tracking.
        """
        return QueryContext(
            original_query=query,
            processed_query=query,
            retrieved_chunks=context,
            prompt=self._template.name,
            response=response,
            metadata={
                "model": llm_response.model,
                "tokens": llm_response.total_tokens,
                "latency_ms": llm_response.latency_ms,
                "finish_reason": llm_response.finish_reason,
            },
        )
