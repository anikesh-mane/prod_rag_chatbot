"""RAG-specific prompt templates."""

import re
from typing import Any

from jinja2 import Environment, BaseLoader, StrictUndefined

from llm.templates.base import BasePromptTemplate


class JinjaPromptTemplate(BasePromptTemplate):
    """Prompt template using Jinja2 for flexible formatting."""

    def __init__(self, name: str, template: str):
        """Initialize Jinja template.

        Args:
            name: Template identifier.
            template: Jinja2 template string.
        """
        self._name = name
        self._template = template
        self._env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._compiled = self._env.from_string(template)

    @property
    def name(self) -> str:
        return self._name

    @property
    def template(self) -> str:
        return self._template

    def format(self, **kwargs: Any) -> str:
        """Format template with Jinja2.

        Args:
            **kwargs: Template variables.

        Returns:
            Formatted prompt.
        """
        return self._compiled.render(**kwargs)

    def get_required_variables(self) -> list[str]:
        """Extract required variables from Jinja template.

        Returns:
            List of variable names.
        """
        # Simple regex to find {{ variable }} patterns
        pattern = r"\{\{\s*(\w+)"
        matches = re.findall(pattern, self._template)
        return list(set(matches))


# =============================================================================
# RAG PROMPT TEMPLATES
# =============================================================================

RAG_QA_TEMPLATE = JinjaPromptTemplate(
    name="rag_qa",
    template="""You are a helpful assistant for Channel Sales and Operations. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.

## Context
{% for doc in context %}
---
Source: {{ doc.source }}
{{ doc.content }}
{% endfor %}
---

## User Question
{{ question }}

## Instructions
1. Answer based ONLY on the context above
2. If the answer is not in the context, say "I don't have enough information to answer this question"
3. Cite the source when possible
4. Be concise and direct

## Answer""",
)


RAG_QA_WITH_HISTORY_TEMPLATE = JinjaPromptTemplate(
    name="rag_qa_with_history",
    template="""You are a helpful assistant for Channel Sales and Operations. Answer the user's question based on the provided context and conversation history.

## Context
{% for doc in context %}
---
Source: {{ doc.source }}
{{ doc.content }}
{% endfor %}
---

## Conversation History
{% for msg in history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}

## Current Question
{{ question }}

## Instructions
1. Answer based ONLY on the context above
2. Consider the conversation history for context
3. If the answer is not in the context, say "I don't have enough information to answer this question"
4. Be concise and direct

## Answer""",
)


SUMMARIZATION_TEMPLATE = JinjaPromptTemplate(
    name="summarization",
    template="""Summarize the following text concisely while preserving key information.

## Text to Summarize
{{ text }}

## Instructions
1. Keep the summary under {{ max_words }} words
2. Preserve key facts and figures
3. Maintain the original tone

## Summary""",
)


QUERY_REWRITE_TEMPLATE = JinjaPromptTemplate(
    name="query_rewrite",
    template="""Rewrite the following user query to be more suitable for semantic search. Make it clear, specific, and search-friendly.

## Original Query
{{ query }}

{% if context %}
## Conversation Context
{{ context }}
{% endif %}

## Instructions
1. Expand abbreviations if clear from context
2. Make implicit information explicit
3. Keep the core intent
4. Output ONLY the rewritten query, nothing else

## Rewritten Query""",
)


# Template registry
TEMPLATES: dict[str, JinjaPromptTemplate] = {
    "rag_qa": RAG_QA_TEMPLATE,
    "rag_qa_with_history": RAG_QA_WITH_HISTORY_TEMPLATE,
    "summarization": SUMMARIZATION_TEMPLATE,
    "query_rewrite": QUERY_REWRITE_TEMPLATE,
}


def get_template(name: str) -> JinjaPromptTemplate:
    """Get a template by name.

    Args:
        name: Template name.

    Returns:
        Template instance.

    Raises:
        KeyError: If template not found.
    """
    if name not in TEMPLATES:
        raise KeyError(f"Template not found: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


def register_template(template: JinjaPromptTemplate) -> None:
    """Register a custom template.

    Args:
        template: Template to register.
    """
    TEMPLATES[template.name] = template
