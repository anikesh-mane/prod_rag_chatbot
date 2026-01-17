"""Base prompt template interface."""

from abc import ABC, abstractmethod
from typing import Any


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return template name."""
        pass

    @property
    @abstractmethod
    def template(self) -> str:
        """Return the template string."""
        pass

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the template with given variables.

        Args:
            **kwargs: Template variables.

        Returns:
            Formatted prompt string.
        """
        pass

    @abstractmethod
    def get_required_variables(self) -> list[str]:
        """Return list of required template variables.

        Returns:
            List of variable names.
        """
        pass
