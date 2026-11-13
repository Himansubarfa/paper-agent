"""Hybrid routing system matching regex heuristics and LLM classification."""

import re
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from src.errors import NeedsClarification
from src.llm.structured import run_structured
from src.logging_setup import get_logger
from src.models.schemas import Intent, IntentResult
from src.orchestration.session import VivaState
from src.prompts import load_prompt

logger = get_logger(__name__)

RULES: dict[Intent, re.Pattern] = {
    "viva_practice": re.compile(
        r"\b(viva|quiz me|examiner|mock (exam|interview)|practice questions?)\b",
        re.IGNORECASE,
    ),
    "generate_slides": re.compile(
        r"\b(ppt|pptx|slides?|deck|presentation|powerpoint)\b",
        re.IGNORECASE,
    ),
    "summarize": re.compile(
        r"\b(summar(y|ise|ize)|tl;?dr|overview of (the )?paper)\b",
        re.IGNORECASE,
    ),
    "extract_concepts": re.compile(
        r"\b(contributions?|key concepts?|methods? used|datasets?|limitations?|extract)\b",
        re.IGNORECASE,
    ),
}

VIVA_STOP_PATTERN = re.compile(r"\b(stop|quit|end viva|exit)\b", re.IGNORECASE)


class HybridRouter:
    """Hybrid intent router combining fast regex rules with an LLM fallback classifier."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self._prompt = load_prompt("router")

    def route(
        self,
        message: str,
        active_task: Optional[Intent] = None,
        ui_selected: Optional[Intent] = None,
        viva_state: Optional[VivaState | str] = None,
        session: Any = None,
    ) -> IntentResult:
        """Route user message to a target intent.

        Follows the strict priority order from Section 8.11:
        1. Viva active mode (bypasses LLM unless stop is issued).
        2. UI selected intent (unless conflicting rule matches).
        3. Regex heuristics in priority order (viva -> slides -> summarize -> concepts).
           Multiple matches fall through to LLM.
        4. Sticky follow-up (<= 6 words when active_task is set and no rule matched).
        5. LLM fallback classifier via run_structured.

        Args:
            message: Raw user query or prompt.
            active_task: The currently active task from session state, if any.
            ui_selected: Explicitly clicked button or UI selector intent, if any.
            viva_state: Current viva examination state, if any.
            session: Optional SessionState instance providing active_task and viva_state.

        Returns:
            IntentResult with classified intent, confidence score, and source.

        Raises:
            NeedsClarification: If LLM confidence < 0.6.
        """
        # Resolve session fields if provided
        if session is not None:
            if active_task is None and hasattr(session, "active_task"):
                active_task = session.active_task
            if viva_state is None and hasattr(session, "viva_state"):
                viva_state = session.viva_state

        clean_msg = message.strip()
        is_stop_word = bool(VIVA_STOP_PATTERN.search(clean_msg))

        # 5. During active viva (viva_state not in {IDLE, FINISHED})
        viva_val = viva_state.value if hasattr(viva_state, "value") else str(viva_state or "idle")
        is_viva_active = viva_val not in (VivaState.IDLE.value, VivaState.FINISHED.value, "idle", "finished", "None")

        if is_viva_active:
            if not is_stop_word:
                return IntentResult(
                    intent=active_task or "viva_practice",
                    confidence=1.0,
                    reasoning="Active viva answer submission",
                    source="rule",
                )
            else:
                # Exits viva mode; routing proceeds normally
                active_task = None

        # Check matched rules in priority order
        matched_rules = [intent for intent, pattern in RULES.items() if pattern.search(clean_msg)]

        # 1. UI Selection Check
        if ui_selected is not None:
            conflicting_rules = [intent for intent in matched_rules if intent != ui_selected]
            if not conflicting_rules:
                return IntentResult(
                    intent=ui_selected,
                    confidence=1.0,
                    reasoning="UI selection",
                    source="ui",
                )

        # 2. Rule matching
        if len(matched_rules) == 1:
            matched_intent = matched_rules[0]
            return IntentResult(
                intent=matched_intent,
                confidence=0.9,
                reasoning=f"Matched keyword rule for {matched_intent}",
                source="rule",
            )
        elif len(matched_rules) > 1:
            # Fall through to LLM classifier with disambiguation note
            logger.info("Multiple intents matched %s; falling through to LLM", matched_rules)
            llm_result = run_structured(
                chain_input={"message": clean_msg},
                prompt=self._prompt,
                schema=IntentResult,
                llm=self.llm,
            )
            note = f"Multiple rules matched ({', '.join(matched_rules)})"
            llm_result.reasoning = f"{note}. {llm_result.reasoning}" if llm_result.reasoning else note
            llm_result.source = "llm"
            if llm_result.confidence < 0.6:
                raise NeedsClarification("Please pick: summary, concepts, slides, viva, or ask a question.")
            return llm_result

        # 4. Sticky follow-up check (no rule matched, short follow-up <= 6 words)
        words = clean_msg.split()
        if active_task is not None and len(words) <= 6 and not is_stop_word:
            return IntentResult(
                intent=active_task,
                confidence=0.8,
                reasoning="sticky follow-up",
                source="rule",
            )

        # 3. LLM classifier fallback
        llm_result = run_structured(
            chain_input={"message": clean_msg},
            prompt=self._prompt,
            schema=IntentResult,
            llm=self.llm,
        )
        llm_result.source = "llm"
        if llm_result.confidence < 0.6:
            raise NeedsClarification("Please pick: summary, concepts, slides, viva, or ask a question.")

        return llm_result
