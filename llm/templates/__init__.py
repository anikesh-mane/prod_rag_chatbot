"""Prompt templates for LLM interactions."""

from llm.templates.base import BasePromptTemplate
from llm.templates.rag_templates import (
    JinjaPromptTemplate,
    RAG_QA_TEMPLATE,
    RAG_QA_WITH_HISTORY_TEMPLATE,
    QUERY_REWRITE_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    get_template,
    register_template,
)

__all__ = [
    "BasePromptTemplate",
    "JinjaPromptTemplate",
    "RAG_QA_TEMPLATE",
    "RAG_QA_WITH_HISTORY_TEMPLATE",
    "QUERY_REWRITE_TEMPLATE",
    "SUMMARIZATION_TEMPLATE",
    "get_template",
    "register_template",
]
