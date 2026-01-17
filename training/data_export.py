"""Training data export from feedback.

Converts user feedback into training datasets for:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Retrieval model fine-tuning
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import structlog

logger = structlog.get_logger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSONL = "jsonl"  # JSON Lines (OpenAI fine-tuning format)
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"  # HuggingFace datasets format


class DatasetType(str, Enum):
    """Types of training datasets."""

    SFT = "sft"  # Supervised fine-tuning
    DPO = "dpo"  # Direct preference optimization
    RETRIEVAL = "retrieval"  # Retrieval model training
    REWARD = "reward"  # Reward model training


@dataclass
class FeedbackRecord:
    """A single feedback record from the database."""

    feedback_id: str
    query_id: str
    user_id: str | None
    query: str
    response: str
    rating: int  # 1-5
    feedback_type: str  # "rating", "correction", "flag"
    correction: str | None  # User-provided correction
    context_chunks: list[dict[str, Any]]  # Retrieved chunks
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTExample:
    """Single example for supervised fine-tuning."""

    messages: list[dict[str, str]]  # OpenAI message format
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI fine-tuning format."""
        return {"messages": self.messages}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "messages": self.messages,
            "metadata": self.metadata,
        }


@dataclass
class DPOExample:
    """Single example for DPO training."""

    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Non-preferred response
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalExample:
    """Single example for retrieval model training."""

    query: str
    positive_passages: list[str]  # Relevant passages
    negative_passages: list[str]  # Non-relevant passages
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "positive_passages": self.positive_passages,
            "negative_passages": self.negative_passages,
            "metadata": self.metadata,
        }


class TrainingDataExporter:
    """Exports feedback data to training datasets.

    Supports multiple export formats and dataset types.
    Implements filtering and quality thresholds.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/training",
        min_rating_for_positive: int = 4,
        max_rating_for_negative: int = 2,
    ):
        """Initialize exporter.

        Args:
            output_dir: Directory for exported datasets.
            min_rating_for_positive: Minimum rating for positive examples.
            max_rating_for_negative: Maximum rating for negative examples.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_rating_for_positive = min_rating_for_positive
        self.max_rating_for_negative = max_rating_for_negative

    def _build_system_prompt(self) -> str:
        """Build system prompt for SFT training."""
        return """You are a helpful assistant that answers questions based on the provided context.
Always ground your answers in the given information. If the context doesn't contain enough information
to answer the question, acknowledge this limitation."""

    def _build_context_string(self, chunks: list[dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("source", "Unknown")
            context_parts.append(f"[{i}] {content}\nSource: {source}")

        return "\n\n".join(context_parts)

    def feedback_to_sft(self, feedback: FeedbackRecord) -> SFTExample | None:
        """Convert feedback to SFT example.

        Only converts high-rating feedback or corrections.

        Args:
            feedback: Feedback record.

        Returns:
            SFT example or None if not suitable.
        """
        # Use correction if available, otherwise use original response for good ratings
        if feedback.correction:
            response = feedback.correction
        elif feedback.rating >= self.min_rating_for_positive:
            response = feedback.response
        else:
            return None

        context = self._build_context_string(feedback.context_chunks)

        user_content = f"""Context:
{context}

Question: {feedback.query}"""

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]

        return SFTExample(
            messages=messages,
            metadata={
                "feedback_id": feedback.feedback_id,
                "query_id": feedback.query_id,
                "rating": feedback.rating,
                "was_corrected": feedback.correction is not None,
            },
        )

    def feedback_pair_to_dpo(
        self,
        positive: FeedbackRecord,
        negative: FeedbackRecord,
    ) -> DPOExample | None:
        """Convert feedback pair to DPO example.

        Args:
            positive: High-rated feedback.
            negative: Low-rated feedback for same/similar query.

        Returns:
            DPO example or None if not suitable.
        """
        if positive.rating < self.min_rating_for_positive:
            return None
        if negative.rating > self.max_rating_for_negative:
            return None

        context = self._build_context_string(positive.context_chunks)

        prompt = f"""Context:
{context}

Question: {positive.query}"""

        # Use correction if available for chosen response
        chosen = positive.correction or positive.response
        rejected = negative.response

        return DPOExample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={
                "positive_feedback_id": positive.feedback_id,
                "negative_feedback_id": negative.feedback_id,
                "positive_rating": positive.rating,
                "negative_rating": negative.rating,
            },
        )

    def feedback_to_retrieval(
        self, feedback: FeedbackRecord
    ) -> RetrievalExample | None:
        """Convert feedback to retrieval training example.

        High-rated responses indicate relevant retrieved passages.

        Args:
            feedback: Feedback record.

        Returns:
            Retrieval example or None if not suitable.
        """
        if not feedback.context_chunks:
            return None

        if feedback.rating >= self.min_rating_for_positive:
            # Good rating = passages were relevant
            positive_passages = [
                chunk.get("content", "")
                for chunk in feedback.context_chunks
                if chunk.get("content")
            ]
            negative_passages = []  # Would need hard negatives from elsewhere
        elif feedback.rating <= self.max_rating_for_negative:
            # Bad rating = passages may not be relevant
            positive_passages = []
            negative_passages = [
                chunk.get("content", "")
                for chunk in feedback.context_chunks
                if chunk.get("content")
            ]
        else:
            return None

        if not positive_passages and not negative_passages:
            return None

        return RetrievalExample(
            query=feedback.query,
            positive_passages=positive_passages,
            negative_passages=negative_passages,
            metadata={
                "feedback_id": feedback.feedback_id,
                "rating": feedback.rating,
            },
        )

    def export_sft_dataset(
        self,
        feedback_records: list[FeedbackRecord],
        format: ExportFormat = ExportFormat.JSONL,
        filename: str | None = None,
    ) -> Path:
        """Export feedback as SFT dataset.

        Args:
            feedback_records: List of feedback records.
            format: Export format.
            filename: Optional filename override.

        Returns:
            Path to exported file.
        """
        examples = []
        for record in feedback_records:
            example = self.feedback_to_sft(record)
            if example:
                examples.append(example)

        logger.info(
            "Converted feedback to SFT examples",
            total=len(feedback_records),
            exported=len(examples),
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = filename or f"sft_dataset_{timestamp}"

        return self._write_dataset(examples, format, filename, DatasetType.SFT)

    def export_dpo_dataset(
        self,
        feedback_pairs: list[tuple[FeedbackRecord, FeedbackRecord]],
        format: ExportFormat = ExportFormat.JSONL,
        filename: str | None = None,
    ) -> Path:
        """Export feedback pairs as DPO dataset.

        Args:
            feedback_pairs: List of (positive, negative) feedback pairs.
            format: Export format.
            filename: Optional filename override.

        Returns:
            Path to exported file.
        """
        examples = []
        for positive, negative in feedback_pairs:
            example = self.feedback_pair_to_dpo(positive, negative)
            if example:
                examples.append(example)

        logger.info(
            "Converted feedback pairs to DPO examples",
            total=len(feedback_pairs),
            exported=len(examples),
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = filename or f"dpo_dataset_{timestamp}"

        return self._write_dataset(examples, format, filename, DatasetType.DPO)

    def export_retrieval_dataset(
        self,
        feedback_records: list[FeedbackRecord],
        format: ExportFormat = ExportFormat.JSONL,
        filename: str | None = None,
    ) -> Path:
        """Export feedback as retrieval training dataset.

        Args:
            feedback_records: List of feedback records.
            format: Export format.
            filename: Optional filename override.

        Returns:
            Path to exported file.
        """
        examples = []
        for record in feedback_records:
            example = self.feedback_to_retrieval(record)
            if example:
                examples.append(example)

        logger.info(
            "Converted feedback to retrieval examples",
            total=len(feedback_records),
            exported=len(examples),
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = filename or f"retrieval_dataset_{timestamp}"

        return self._write_dataset(examples, format, filename, DatasetType.RETRIEVAL)

    def _write_dataset(
        self,
        examples: list[SFTExample | DPOExample | RetrievalExample],
        format: ExportFormat,
        filename: str,
        dataset_type: DatasetType,
    ) -> Path:
        """Write dataset to file.

        Args:
            examples: List of examples.
            format: Export format.
            filename: Base filename.
            dataset_type: Type of dataset.

        Returns:
            Path to written file.
        """
        if format == ExportFormat.JSONL:
            return self._write_jsonl(examples, filename)
        elif format == ExportFormat.CSV:
            return self._write_csv(examples, filename, dataset_type)
        elif format == ExportFormat.PARQUET:
            return self._write_parquet(examples, filename)
        elif format == ExportFormat.HUGGINGFACE:
            return self._write_huggingface(examples, filename, dataset_type)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _write_jsonl(
        self,
        examples: list[SFTExample | DPOExample | RetrievalExample],
        filename: str,
    ) -> Path:
        """Write JSONL format."""
        filepath = self.output_dir / f"{filename}.jsonl"

        with open(filepath, "w", encoding="utf-8") as f:
            for example in examples:
                if isinstance(example, SFTExample):
                    # Use OpenAI format for SFT
                    line = json.dumps(example.to_openai_format(), ensure_ascii=False)
                else:
                    line = json.dumps(example.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        logger.info("Wrote JSONL dataset", path=str(filepath), count=len(examples))
        return filepath

    def _write_csv(
        self,
        examples: list[SFTExample | DPOExample | RetrievalExample],
        filename: str,
        dataset_type: DatasetType,
    ) -> Path:
        """Write CSV format."""
        filepath = self.output_dir / f"{filename}.csv"

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            if dataset_type == DatasetType.SFT:
                self._write_sft_csv(f, examples)  # type: ignore
            elif dataset_type == DatasetType.DPO:
                self._write_dpo_csv(f, examples)  # type: ignore
            elif dataset_type == DatasetType.RETRIEVAL:
                self._write_retrieval_csv(f, examples)  # type: ignore

        logger.info("Wrote CSV dataset", path=str(filepath), count=len(examples))
        return filepath

    def _write_sft_csv(self, f: TextIO, examples: list[SFTExample]) -> None:
        """Write SFT examples to CSV."""
        writer = csv.writer(f)
        writer.writerow(["system", "user", "assistant"])

        for ex in examples:
            system = next(
                (m["content"] for m in ex.messages if m["role"] == "system"), ""
            )
            user = next((m["content"] for m in ex.messages if m["role"] == "user"), "")
            assistant = next(
                (m["content"] for m in ex.messages if m["role"] == "assistant"), ""
            )
            writer.writerow([system, user, assistant])

    def _write_dpo_csv(self, f: TextIO, examples: list[DPOExample]) -> None:
        """Write DPO examples to CSV."""
        writer = csv.writer(f)
        writer.writerow(["prompt", "chosen", "rejected"])

        for ex in examples:
            writer.writerow([ex.prompt, ex.chosen, ex.rejected])

    def _write_retrieval_csv(self, f: TextIO, examples: list[RetrievalExample]) -> None:
        """Write retrieval examples to CSV."""
        writer = csv.writer(f)
        writer.writerow(["query", "positive_passages", "negative_passages"])

        for ex in examples:
            writer.writerow([
                ex.query,
                json.dumps(ex.positive_passages),
                json.dumps(ex.negative_passages),
            ])

    def _write_parquet(
        self,
        examples: list[SFTExample | DPOExample | RetrievalExample],
        filename: str,
    ) -> Path:
        """Write Parquet format."""
        filepath = self.output_dir / f"{filename}.parquet"

        try:
            import pandas as pd

            data = [ex.to_dict() for ex in examples]
            df = pd.DataFrame(data)
            df.to_parquet(filepath, index=False)

            logger.info("Wrote Parquet dataset", path=str(filepath), count=len(examples))
        except ImportError:
            logger.warning("pandas/pyarrow not installed, falling back to JSONL")
            return self._write_jsonl(examples, filename)

        return filepath

    def _write_huggingface(
        self,
        examples: list[SFTExample | DPOExample | RetrievalExample],
        filename: str,
        dataset_type: DatasetType,
    ) -> Path:
        """Write HuggingFace datasets format."""
        filepath = self.output_dir / filename

        try:
            from datasets import Dataset

            data = [ex.to_dict() for ex in examples]
            dataset = Dataset.from_list(data)
            dataset.save_to_disk(str(filepath))

            logger.info(
                "Wrote HuggingFace dataset", path=str(filepath), count=len(examples)
            )
        except ImportError:
            logger.warning("datasets not installed, falling back to JSONL")
            return self._write_jsonl(examples, filename)

        return filepath

    def get_export_stats(self, feedback_records: list[FeedbackRecord]) -> dict:
        """Get statistics about exportable data.

        Args:
            feedback_records: List of feedback records.

        Returns:
            Statistics dictionary.
        """
        total = len(feedback_records)
        high_rated = sum(
            1 for r in feedback_records if r.rating >= self.min_rating_for_positive
        )
        low_rated = sum(
            1 for r in feedback_records if r.rating <= self.max_rating_for_negative
        )
        with_corrections = sum(1 for r in feedback_records if r.correction)
        with_context = sum(1 for r in feedback_records if r.context_chunks)

        return {
            "total_records": total,
            "high_rated": high_rated,
            "low_rated": low_rated,
            "with_corrections": with_corrections,
            "with_context": with_context,
            "sft_exportable": high_rated + with_corrections,
            "dpo_pairs_possible": min(high_rated, low_rated),
            "retrieval_exportable": with_context,
        }
