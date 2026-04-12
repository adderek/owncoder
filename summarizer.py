import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

async def summarize_session_background(
    config: "Config",
    client: "AsyncOpenAI",
    messages: list[dict],
    on_summary_ready: callable | None = None,
) -> None:
    \"\"\"
    Background task to summarize the conversation history.
    This is meant to be spawned and not awaited in the main loop.
    \"\"\"
    if not config.ui.q_summaries:
        return

    # We want to avoid summarizing if the history is too short
    # or if we just did a compaction.
    if len(messages) < 10:
        return

    try:
        # 1. Prepare the summarization prompt
        # We only want to summarize the messages that aren't the most recent ones
        # to keep the context relevant, but for a "Q-summary" we might want
        # a more holistic view.
        
        # Let's take the history excluding the last few messages
        # so we don't summarize what the user just said or what the assistant just responded to.
        summary_messages = messages[:-5]
        
        if not summary_messages:
            return

        # 2. Call the LLM to generate a summary
        # We use a specific system prompt for summarization.
        summary_system_prompt = (
            "You are a helpful assistant that summarizes conversation history. "
            "Provide a concise, bulleted summary of the key points, decisions made, "
            "and tasks completed in this session. Focus on technical details and progress."
        )

        api_messages = [
            {"role": "system", "content": summary_system_prompt},
            *summary_messages
        ]

        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=api_messages,
            max_tokens=500,
        )

        summary_text = response.choices[0].message.content
        if not summary_text:
            return

        # 3. Notify the UI or update the session
        if on_summary_ready:
            # The callback might be a simple function that updates the session object
            # or it might be a coroutine that handles more complex logic.
            if asyncio.iscoroutinefunction(on_summary_ready):
                await on_summary_ready(summary_text)
            else:
                on_summary_ready(summary_text)

    except Exception as e:
        logger.error(f"Error in background summarization: {e}", exc_info=True)

def _count_tokens_approx(text: str) -> int:
    # Placeholder for the actual token counting logic used in agent.py
    # In a real implementation, this would use tiktoken.
    return len(text) // 4
