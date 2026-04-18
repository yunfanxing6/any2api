import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EMBEDDED_QWEN_ROOT = ROOT / "embedded_qwen"

if str(EMBEDDED_QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(EMBEDDED_QWEN_ROOT))

from app.dataplane.reverse.protocol.tool_parser import ParsedToolCall
from app.platform.tokens import estimate_prompt_tokens, estimate_tokens, estimate_tool_call_tokens
from app.products.openai._format import build_usage
from app.products.openai.chat import _build_stream_usage
from backend.services.openai_stream_translator import OpenAIStreamTranslator


def test_build_stream_usage_matches_text_completion_path() -> None:
    message = "hello world"
    answer_text = "streamed answer"
    thinking_text = "reasoning"

    usage = _build_stream_usage(message, answer_text=answer_text, thinking_text=thinking_text)

    expected = build_usage(
        estimate_prompt_tokens(message),
        estimate_tokens(answer_text) + estimate_tokens(thinking_text),
        reasoning_tokens=estimate_tokens(thinking_text),
    )
    assert usage == expected


def test_build_stream_usage_matches_tool_call_path() -> None:
    message = "call a tool"
    thinking_text = "tool reasoning"
    tool_calls = [ParsedToolCall(call_id="call_1", name="search", arguments='{"q":"hi"}')]

    usage = _build_stream_usage(message, thinking_text=thinking_text, tool_calls=tool_calls)

    expected = build_usage(
        estimate_prompt_tokens(message),
        estimate_tool_call_tokens(tool_calls) + estimate_tokens(thinking_text),
        reasoning_tokens=estimate_tokens(thinking_text),
    )
    assert usage == expected


def test_qwen_stream_translator_final_chunk_includes_usage() -> None:
    translator = OpenAIStreamTranslator(
        completion_id="chatcmpl_test",
        created=123,
        model_name="qwen3.5-plus",
        client_profile="default",
    )

    chunks = translator.finalize(
        "stop",
        usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    )

    final_payload = json.loads(chunks[-2][len("data: ") : -2])
    assert final_payload["usage"] == {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    assert final_payload["choices"][0]["finish_reason"] == "stop"
