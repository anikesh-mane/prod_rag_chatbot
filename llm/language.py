"""Language detection and translation services.

Uses fastText for language detection and supports multiple translation backends.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import fasttext

logger = structlog.get_logger(__name__)


class SupportedLanguage(str, Enum):
    """Languages supported by the system."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


# Map fastText language codes to our supported languages
FASTTEXT_TO_SUPPORTED: dict[str, SupportedLanguage] = {
    "__label__en": SupportedLanguage.ENGLISH,
    "__label__es": SupportedLanguage.SPANISH,
    "__label__fr": SupportedLanguage.FRENCH,
    "__label__de": SupportedLanguage.GERMAN,
    "__label__it": SupportedLanguage.ITALIAN,
    "__label__pt": SupportedLanguage.PORTUGUESE,
    "__label__nl": SupportedLanguage.DUTCH,
    "__label__ru": SupportedLanguage.RUSSIAN,
    "__label__zh": SupportedLanguage.CHINESE,
    "__label__ja": SupportedLanguage.JAPANESE,
    "__label__ko": SupportedLanguage.KOREAN,
    "__label__ar": SupportedLanguage.ARABIC,
    "__label__hi": SupportedLanguage.HINDI,
}


class LanguageDetector:
    """Detect text language using fastText.

    Uses the lid.176.bin model for 176-language detection.
    Falls back to English if detection fails or confidence is low.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        default_language: SupportedLanguage = SupportedLanguage.ENGLISH,
    ):
        """Initialize language detector.

        Args:
            model_path: Path to fastText model file.
            confidence_threshold: Minimum confidence for detection.
            default_language: Fallback language.
        """
        self._model: fasttext.FastText._FastText | None = None
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._default_language = default_language
        self._loaded = False

    def _load_model(self) -> None:
        """Lazy load the fastText model."""
        if self._loaded:
            return

        try:
            import fasttext

            # Suppress fastText warnings about deprecated model
            fasttext.FastText.eprint = lambda x: None

            if self._model_path and Path(self._model_path).exists():
                self._model = fasttext.load_model(str(self._model_path))
            else:
                # Try common locations
                common_paths = [
                    Path("models/lid.176.bin"),
                    Path("/app/models/lid.176.bin"),
                    Path.home() / ".cache" / "fasttext" / "lid.176.bin",
                ]
                for path in common_paths:
                    if path.exists():
                        self._model = fasttext.load_model(str(path))
                        break

            if self._model:
                self._loaded = True
                logger.info("FastText language model loaded")
            else:
                logger.warning(
                    "FastText model not found, using default language",
                    default=self._default_language.value,
                )

        except ImportError:
            logger.warning("fasttext not installed, language detection disabled")
        except Exception as e:
            logger.error("Failed to load language model", error=str(e))

    def detect(self, text: str) -> tuple[SupportedLanguage, float]:
        """Detect the language of text.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (detected language, confidence score).
        """
        if not text or not text.strip():
            return self._default_language, 1.0

        self._load_model()

        if not self._model:
            return self._default_language, 0.0

        try:
            # Clean text for detection (single line, no newlines)
            clean_text = " ".join(text.split())[:500]  # Limit length

            predictions = self._model.predict(clean_text, k=1)
            label = predictions[0][0]
            confidence = float(predictions[1][0])

            if confidence < self._confidence_threshold:
                logger.debug(
                    "Low confidence detection",
                    detected=label,
                    confidence=confidence,
                    using_default=True,
                )
                return self._default_language, confidence

            # Map to supported language
            if label in FASTTEXT_TO_SUPPORTED:
                detected = FASTTEXT_TO_SUPPORTED[label]
                logger.debug(
                    "Language detected",
                    language=detected.value,
                    confidence=confidence,
                )
                return detected, confidence
            else:
                # Unsupported language, use default
                logger.debug(
                    "Unsupported language detected",
                    detected=label,
                    confidence=confidence,
                )
                return self._default_language, confidence

        except Exception as e:
            logger.error("Language detection failed", error=str(e))
            return self._default_language, 0.0

    def is_english(self, text: str) -> bool:
        """Check if text is in English.

        Args:
            text: Text to check.

        Returns:
            True if text is English with sufficient confidence.
        """
        lang, conf = self.detect(text)
        return lang == SupportedLanguage.ENGLISH and conf >= self._confidence_threshold


class BaseTranslator(ABC):
    """Base class for translation services."""

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: SupportedLanguage,
        target_lang: SupportedLanguage,
    ) -> str:
        """Translate text between languages.

        Args:
            text: Text to translate.
            source_lang: Source language.
            target_lang: Target language.

        Returns:
            Translated text.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if translator is available."""
        pass


class OpenAITranslator(BaseTranslator):
    """Translation using OpenAI's GPT models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ):
        """Initialize OpenAI translator.

        Args:
            api_key: OpenAI API key.
            model: Model to use for translation.
        """
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def translate(
        self,
        text: str,
        source_lang: SupportedLanguage,
        target_lang: SupportedLanguage,
    ) -> str:
        """Translate text using OpenAI.

        Args:
            text: Text to translate.
            source_lang: Source language.
            target_lang: Target language.

        Returns:
            Translated text.
        """
        if source_lang == target_lang:
            return text

        prompt = f"""Translate the following text from {source_lang.name.title()} to {target_lang.name.title()}.
Only output the translation, no explanations.

Text: {text}"""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=len(text) * 2,  # Allow for expansion
            )

            translated = response.choices[0].message.content or text
            logger.info(
                "Translation complete",
                source=source_lang.value,
                target=target_lang.value,
                input_len=len(text),
                output_len=len(translated),
            )
            return translated.strip()

        except Exception as e:
            logger.error("Translation failed", error=str(e))
            return text  # Return original on failure

    async def health_check(self) -> bool:
        """Check OpenAI API availability."""
        try:
            await self._client.models.retrieve(self._model)
            return True
        except Exception:
            return False


class NoOpTranslator(BaseTranslator):
    """No-op translator that returns text unchanged.

    Used when translation is disabled or unavailable.
    """

    async def translate(
        self,
        text: str,
        source_lang: SupportedLanguage,
        target_lang: SupportedLanguage,
    ) -> str:
        """Return text unchanged."""
        return text

    async def health_check(self) -> bool:
        """Always healthy."""
        return True


class LanguageService:
    """Unified service for language detection and translation.

    Handles the full flow:
    1. Detect input language
    2. Translate to English for processing (if needed)
    3. Translate response back to user's language (if needed)
    """

    def __init__(
        self,
        detector: LanguageDetector | None = None,
        translator: BaseTranslator | None = None,
        processing_language: SupportedLanguage = SupportedLanguage.ENGLISH,
    ):
        """Initialize language service.

        Args:
            detector: Language detector instance.
            translator: Translator instance.
            processing_language: Language used for RAG processing.
        """
        self._detector = detector or LanguageDetector()
        self._translator = translator or NoOpTranslator()
        self._processing_language = processing_language

    async def prepare_query(
        self, query: str
    ) -> tuple[str, SupportedLanguage, float]:
        """Prepare query for processing.

        Detects language and translates to processing language if needed.

        Args:
            query: User query.

        Returns:
            Tuple of (processed query, original language, detection confidence).
        """
        detected_lang, confidence = self._detector.detect(query)

        if detected_lang == self._processing_language:
            return query, detected_lang, confidence

        translated = await self._translator.translate(
            text=query,
            source_lang=detected_lang,
            target_lang=self._processing_language,
        )

        logger.info(
            "Query translated for processing",
            original_lang=detected_lang.value,
            target_lang=self._processing_language.value,
        )

        return translated, detected_lang, confidence

    async def prepare_response(
        self,
        response: str,
        target_lang: SupportedLanguage,
    ) -> str:
        """Prepare response for user.

        Translates response to user's language if different from processing language.

        Args:
            response: Response in processing language.
            target_lang: User's language.

        Returns:
            Response in user's language.
        """
        if target_lang == self._processing_language:
            return response

        translated = await self._translator.translate(
            text=response,
            source_lang=self._processing_language,
            target_lang=target_lang,
        )

        logger.info(
            "Response translated for user",
            source_lang=self._processing_language.value,
            target_lang=target_lang.value,
        )

        return translated


# Factory function for creating language service
def get_language_service(
    openai_api_key: str | None = None,
    translation_enabled: bool = True,
) -> LanguageService:
    """Create a configured language service.

    Args:
        openai_api_key: OpenAI API key for translation.
        translation_enabled: Whether to enable translation.

    Returns:
        Configured language service.
    """
    detector = LanguageDetector()

    if translation_enabled and openai_api_key:
        translator: BaseTranslator = OpenAITranslator(api_key=openai_api_key)
    else:
        translator = NoOpTranslator()

    return LanguageService(detector=detector, translator=translator)
