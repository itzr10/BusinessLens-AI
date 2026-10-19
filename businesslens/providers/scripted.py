"""A scripted stand-in provider for offline tests and the offline benchmark. No network."""

from __future__ import annotations

from businesslens.providers.base import (
    Conversation, Provider, ToolResult, ToolSpec, Turn,
)


class ScriptedConversation(Conversation):
    def __init__(self, system: str, user_text: str, tools: list[ToolSpec]):
        self.system, self.user_text, self.tools = system, user_text, tools
        self.tool_results: list[list[ToolResult]] = []


class ScriptedProvider(Provider):
    """Plays back ``steps``: each is a ``Turn`` to return or an exception to raise.

    A step may also be a callable taking the conversation and returning a Turn,
    which lets a script build its answer from the tool results it received.
    """

    name = "scripted"

    def __init__(self, *steps):
        self.steps = list(steps)
        self.generate_calls = 0
        self.conversations: list[ScriptedConversation] = []

    def start(self, system, user_text, tools):
        conv = ScriptedConversation(system, user_text, tools)
        self.conversations.append(conv)
        return conv

    def generate(self, conversation):
        self.generate_calls += 1
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(conversation) if callable(step) else step

    def add_tool_results(self, conversation, turn: Turn, results):
        conversation.tool_results.append(results)


# ---- fake vendor SDK clients (used by tests and the benchmark; no network) ----

def fake_openai_client(*steps):
    """Fake ``openai.OpenAI``: each step is a response object or an exception to raise."""
    from types import SimpleNamespace

    queue = list(steps)

    def create(**kwargs):
        step = queue.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def fake_gemini_client(*steps):
    from types import SimpleNamespace

    queue = list(steps)

    def generate_content(model, contents, config):
        step = queue.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
