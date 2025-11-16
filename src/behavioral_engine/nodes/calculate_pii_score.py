from __future__ import annotations

from typing import Any, Dict, Iterable, List


_ANALYZER = None  # Lazy-initialized Presidio AnalyzerEngine


def _ensure_analyzer():
    """Create and cache a Presidio AnalyzerEngine with spaCy NLP.

    Tries to load `en_core_web_sm` and will attempt a light-weight download
    if missing. Falls back gracefully if Presidio/spaCy are unavailable.
    """
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER

    try:
        # Import here to keep optional dependency lightweight during import time
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore
        import spacy  # type: ignore

        # Ensure an English spaCy model is present; prefer small model for footprint
        model_name = "en_core_web_sm"
        try:
            # Try loading as a package first (installed via pip/uv)
            nlp = spacy.load("en_core_web_sm")
            print(f"--- [Behavior Engine] Loaded spaCy model: {model_name}")
        except OSError:
            # Try loading by module name (when installed as a package)
            try:
                import en_core_web_sm  # type: ignore
                nlp = en_core_web_sm.load()
                print(f"--- [Behavior Engine] Loaded spaCy model via import: {model_name}")
            except (ImportError, OSError):
                # Fall back to blank English pipeline (regex-only recognizers will still work)
                print("--- [Behavior Engine] Warning: en_core_web_sm not found, using blank pipeline")
                nlp = spacy.blank("en")
                model_name = "en"

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }

        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        
        # Create analyzer with default English recognizers
        _ANALYZER = AnalyzerEngine(nlp_engine=nlp_engine)
        
        # Get list of supported entities for debugging
        supported = _ANALYZER.get_supported_entities(language="en")
        print(f"--- [Behavior Engine] Presidio initialized with entities: {', '.join(sorted(supported))}")
    except Exception as e:
        # If anything goes wrong (missing deps, etc.), keep analyzer as None
        print(f"--- [Behavior Engine] Failed to initialize analyzer: {e}")
        _ANALYZER = None

    return _ANALYZER


def _extract_text_from_conversation(conversation_history: Iterable[Dict[str, Any]]) -> str:
    """Flatten OpenAI-style conversation history into a single text string.

    Supports messages where `content` is a string or a list of content parts.
    Non-text parts (e.g., images) are ignored.
    """
    chunks: List[str] = []
    for msg in conversation_history or []:
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                chunks.append(str(content))
        elif isinstance(content, list):
            for part in content:
                # OpenAI-like structured parts
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        chunks.append(str(part["text"]))
                    elif "text" in part and isinstance(part["text"], str):
                        chunks.append(str(part["text"]))
                elif isinstance(part, str):
                    chunks.append(part)
        # Ignore other shapes (e.g., tool_calls, images)
    return "\n".join(chunks)


def _compute_pii_score_from_entities(entities: List[Dict[str, Any]]) -> float:
    """Compute a score in [0,1] from detected PII entities.

    Lower score means more sensitive PII present.

    Method:
    - Map entity types to sensitivity weights (risk points).
    - Sum(weight * confidence) across all detections to get total_risk.
    - Convert to score with 1/(1 + total_risk).
    """
    # Sensitivity tiers (can be tuned). Keys are Presidio entity types.
    VERY_HIGH = {
        "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE", "SWIFT_CODE",
        "US_SSN", "US_SSN_TEMP", "CRYPTO", "MEDICAL_LICENSE",
        "AU_ABN", "AU_ACN", "SG_NRIC", "PASSPORT", "US_DRIVER_LICENSE",
    }
    HIGH = {
        "PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS", "MAC_ADDRESS",
        "US_ITIN", "US_TAXPAYER_ID", "UK_NHS", "IMEI", "IMSI",
        "NRP", "NRIC", "BITCOIN_ADDRESS",
    }
    MEDIUM = {
        "PERSON", "AGE", "NRP_DATE", "NRP_EMAIL",
        "ORGANIZATION", "USERNAME",
    }
    LOW = {
        "LOCATION", "URL", "ZIP_CODE", "US_ZIP_CODE","DATE_TIME"
    }

    def weight_for(entity_type: str) -> float:
        if entity_type in VERY_HIGH:
            return 1.0
        if entity_type in HIGH:
            return 0.7
        if entity_type in MEDIUM:
            return 0.4
        if entity_type in LOW:
            return 0.2
        # Unknown types treated as medium risk
        return 0.4

    total_risk = 0.0
    for ent in entities:
        etype = str(ent.get("entity_type") or ent.get("type") or "").upper()
        confidence = float(ent.get("score") or ent.get("confidence") or 1.0)
        total_risk += weight_for(etype) * max(0.0, min(confidence, 1.0))

    # Score in [0,1]; decreases with higher total risk
    return 1.0 / (1.0 + total_risk)


def calculate_pii_score(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate a PII score for a conversation using Microsoft Presidio.

    - Input: state with key `conversation_history` as OpenAI-style list of messages
    - Output: dict with `pii_score` in [0,1]; lower means more sensitive PII present
    """
    conversation_history = state.get("conversation_history", [])
    text = _extract_text_from_conversation(conversation_history)

    print(f"--- [Behavior Engine] Analyzing text: {text[:200]}..." if len(text) > 200 else f"--- [Behavior Engine] Analyzing text: {text}")

    analyzer = _ensure_analyzer()
    entities: List[Dict[str, Any]] = []

    if analyzer is not None and text.strip():
        try:
            results = analyzer.analyze(text=text, language="en")
            # Convert to lightweight dicts for scoring
            for r in results:
                entities.append({
                    "entity_type": getattr(r, "entity_type", None),
                    "score": getattr(r, "score", None),
                    "start": getattr(r, "start", None),
                    "end": getattr(r, "end", None),
                })
            print(f"--- [Behavior Engine] Detected {len(entities)} PII entities: {[e['entity_type'] for e in entities]}")
        except Exception as e:
            # If analysis fails, fall back to score=1 (no penalty)
            print(f"--- [Behavior Engine] Analysis failed: {e}")
            entities = []
    else:
        print(f"--- [Behavior Engine] Analyzer not available or empty text")

    pii_score = _compute_pii_score_from_entities(entities)
    print(f"--- [Behavior Engine] PII Score: {pii_score:.4f} (lower = more sensitive PII) ---")
    return {"pii_score": pii_score}