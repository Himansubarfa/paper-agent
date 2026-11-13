"""Structured output execution with robust fallback strategies."""

from typing import Any, Type, TypeVar
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

from src.errors import AgentError
from src.logging_setup import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _append_instruction(prompt: ChatPromptTemplate, instruction: str) -> ChatPromptTemplate:
    """Return a new ChatPromptTemplate with instruction appended to messages."""
    messages = list(prompt.messages)
    messages.append(("human", instruction))
    return ChatPromptTemplate.from_messages(messages)


def run_structured(
    chain_input: dict[str, Any],
    prompt: ChatPromptTemplate,
    schema: Type[T],
    llm: BaseChatModel,
    retries: int = 1,
) -> T:
    """Execute a structured output chain with fallback and retry mechanisms.

    Strategy in order:
    1. llm.with_structured_output(schema)
    2. On failure, OutputFixingParser with a second LLM call
    3. One final retry with appended JSON-only instruction
    4. On persistent failure, raise AgentError with clear details and last raw output.

    Args:
        chain_input: Dictionary of input variables for the prompt template.
        prompt: ChatPromptTemplate defining system and human instructions.
        schema: Target Pydantic model class for structured response.
        llm: Language model instance implementing BaseChatModel.
        retries: Number of retry attempts for the final prompt adjustment.

    Returns:
        Instance of schema populated from the LLM output.

    Raises:
        AgentError: If all fallback and retry attempts are exhausted.
    """
    last_raw_output: str = ""
    last_error: Exception | None = None

    # Step 1: Attempt direct with_structured_output
    try:
        if isinstance(llm, ChatGroq):
            structured_llm = llm.with_structured_output(schema, method="json_schema")
        else:
            structured_llm = llm.with_structured_output(schema)
        chain = prompt | structured_llm
        result = chain.invoke(chain_input)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        if isinstance(result, BaseModel):
            return schema.model_validate(result.model_dump())
    except Exception as exc1:
        last_error = exc1
        logger.warning(
            "with_structured_output failed for schema %s: %s. Attempting OutputFixingParser fallback.",
            schema.__name__,
            exc1,
        )

    # Step 2: Attempt OutputFixingParser with a second LLM call
    try:
        raw_msg = (prompt | llm).invoke(chain_input)
        last_raw_output = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)

        try:
            from langchain.output_parsers import OutputFixingParser
            fixing_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=schema),
                llm=llm,
            )
            return fixing_parser.parse(last_raw_output)
        except ImportError:
            parser = PydanticOutputParser(pydantic_object=schema)
            return parser.parse(last_raw_output)
    except Exception as exc2:
        last_error = exc2
        logger.warning(
            "OutputFixingParser fallback failed for schema %s: %s. Attempting final retry with appended instruction.",
            schema.__name__,
            exc2,
        )

    # Step 3: Final retry with appended instruction
    appended_prompt = _append_instruction(
        prompt,
        "Return ONLY valid JSON matching the schema. No prose.",
    )

    for attempt in range(max(1, retries)):
        try:
            structured_llm = llm.with_structured_output(schema)
            chain = appended_prompt | structured_llm
            result = chain.invoke(chain_input)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            if isinstance(result, BaseModel):
                return schema.model_validate(result.model_dump())
        except Exception as exc3:
            last_error = exc3
            logger.warning(
                "Final retry %d failed for schema %s: %s",
                attempt + 1,
                schema.__name__,
                exc3,
            )

    raise AgentError(
        f"Failed to produce valid structured output for {schema.__name__}: {last_error}. "
        f"Last raw output: {last_raw_output}"
    )
