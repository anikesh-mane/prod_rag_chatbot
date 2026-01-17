"""Base LLM client interface."""

from abc import ABC, abstractmethod

from schemas import LLMRequest, LLMResponse


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            request: LLM request with prompt and parameters.

        Returns:
            LLM response with content and metadata.
        """
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest):
        """Generate a streaming response from the LLM.

        Args:
            request: LLM request with prompt and parameters.

        Yields:
            Response chunks as they arrive.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM client is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        pass
