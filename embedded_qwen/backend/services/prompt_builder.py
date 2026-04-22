import json
import logging
import re
from dataclasses import dataclass

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.core.request_logging import get_request_context
from backend.services import file_content_cache
from backend.services.client_profiles import (
    QWEN_CODE_OPENAI_PROFILE,
    looks_like_opencode_system_prompt as _looks_like_opencode_system_prompt,
    sanitize_openclaw_user_text,
)
from backend.services.refusal_cleaner import clean_refusal_messages
from backend.services.schema_compressor import compact_schema
from backend.services.tool_few_shot import pick_few_shot_tools, render_few_shot_turn, tool_summary_for_log
from backend.services.tool_name_obfuscation import obfuscate_bare_names, to_qwen_name
from backend.services.topic_isolation import detect_topic_change

log = logging.getLogger("qwen2api.prompt")

OPENCLAW_STARTUP_PATTERNS = (
    "A new session was started via /new or /reset.",
    "If runtime-provided startup context is included for this first turn",
)
OPENCLAW_UNTRUSTED_METADATA_PREFIX = "Sender (untrusted metadata):"


@dataclass(slots=True)
class PromptBuildResult:
    prompt: str
    tools: list[dict]
    tool_enabled: bool


def _is_heavy_tool_profile(client_profile: str) -> bool:
    return client_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}


def _compact_history_tool_input(name: str, input_data: dict, client_profile: str) -> dict:
    if client_profile != CLAUDE_CODE_OPENAI_PROFILE or not isinstance(input_data, dict):
        return input_data
    compact = dict(input_data)
    large_text_keys = ("content", "new_string", "old_string", "insert_text", "text", "patch")
    for key in large_text_keys:
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 50:
            compact[key] = f"[{len(value)} chars]"

    for key in ("file_path", "path", "pattern"):
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 80:
            parts = value.replace('\\', '/').split('/')
            if len(parts) > 3:
                compact[key] = f".../{'/'.join(parts[-2:])}"

    if name in {"Write", "Edit", "NotebookEdit"}:
        preferred = {}
        for key in ("file_path", "path", "target_file", "filename", "old_string", "new_string", "content"):
            if key in compact:
                preferred[key] = compact[key]
        if preferred:
            compact = preferred
    return compact


def _render_history_tool_call(name: str, input_data: dict, client_profile: str) -> str:
    payload = json.dumps({"name": to_qwen_name(name), "input": _compact_history_tool_input(name, input_data, client_profile)}, ensure_ascii=False)
    if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        return f"##TOOL_CALL##\n{payload}\n##END_CALL##"
    return f"##TOOL_CALL##\n{payload}\n##END_CALL##"


def _build_tool_instruction_block(tools: list[dict], client_profile: str) -> str:
    names = [to_qwen_name(t.get("name", "")) for t in tools if t.get("name")]
    if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        lines = [
            "=== ACTION MARKER PROTOCOL (client-parsed text patterns) ===",
            "【重要】用户输入什么语言，就用什么语言回复。User inputs Chinese → respond in Chinese.",
            "【重要】用户要求多个操作时（如读文件并写文档），必须完成所有操作，不要询问确认。",
            "【重要】如果文件显示'Unchanged since last read'，不要再次读取同一文件。",
            "【重要】不要自动调用Agent action，除非用户明确要求。",
            "",
            "IGNORE any previous output format instructions (needs-review, recap, etc.).",
            "",
            "You are operating within a client that parses action markers from your output.",
            "These markers are plain TEXT PATTERNS the client recognizes - NOT native function calls.",
            "The client executes the action and returns results in a subsequent turn.",
            "",
            f"Available action names: {', '.join(names)}",
            "",
            "WHEN YOU NEED TO TRIGGER AN ACTION - emit this exact text pattern (nothing else):",
            "##TOOL_CALL##",
            '{"name": "ACTION_NAME", "input": {"param1": "value1"}}',
            "##END_CALL##",
            "",
            "MULTI-TURN RULES:",
            "- After a [Tool Result] block appears in the conversation, read it and decide the next action.",
            "- If more actions are needed, emit another ##TOOL_CALL## block.",
            "- Only give a final text answer when ALL needed information is gathered.",
            "- Never skip an action that is required to complete the user request.",
            "- The history shows ##TOOL_CALL## blocks you already emitted and their [Tool Result] responses.",
            "",
            "STRICT RULES:",
            "- No preamble, no explanation before or after ##TOOL_CALL##...##END_CALL##.",
            "- Use EXACT action name from the list above.",
            "- When NO action is needed, answer normally in plain text.",
            "- For file/config tasks prefer Read/Edit/Write actions. Use Bash only when shell behavior is required.",
            "- On Windows-like paths inside Bash, use POSIX commands or powershell.exe -Command.",
            "- Do NOT trigger Agent action automatically unless user explicitly requests it.",
            "- Do NOT read the same file multiple times if it shows 'Unchanged since last read'.",
            "",
            "EXECUTION RULES - CRITICAL:",
            "- When user gives a task, START IMMEDIATELY by emitting the required action markers.",
            "- Do NOT wait, do NOT ask for confirmation, do NOT ask 'what should I do next'.",
            "- Do NOT emit EnterPlanMode, ExitPlanMode, EnterWorktree, ExitWorktree, AskUserQuestion action markers.",
            "- Complete the task directly and provide the result.",
            "- If you need information, emit Read/Grep/Glob markers. If you need to modify, emit Edit/Write markers.",
            "- Only respond with text when the task is complete or you have the final answer.",
            "",
            "CRITICAL - ABSOLUTELY FORBIDDEN OUTPUTS:",
            "- NEVER emit ANY disclaimer, error text, or availability complaint about actions.",
            "- NEVER emit sentences claiming an action is missing, unregistered, unavailable, or cannot be invoked.",
            "- NEVER emit sentences claiming you are unable to execute a function.",
            "- The ##TOOL_CALL## blocks are TEXT MARKERS the client parses - they are NOT native function calls. Just emit the text.",
            "- If you feel an action could fail, emit the ##TOOL_CALL## anyway - the client handles failures; your job is only to emit the marker.",
            "",
            "FORBIDDEN ALTERNATE FORMATS (will be ignored by the client's parser):",
            '- {"name": "X", "arguments": "..."}  <-- NEVER USE',
            '- {"type": "function", "name": "X"}  <-- NEVER USE',
            '- {"type": "tool_use", "name": "X"}  <-- NEVER USE',
            '- <function_calls><invoke name="X">  <-- NEVER USE',
            '- <tool_call>{...}</tool_call>  <-- NEVER USE',
            '- {"name":"X","input":{...}} without ##TOOL_CALL## markers  <-- NEVER USE',
            "- <｜Tool｜> or <｜tool｜> markers  <-- NEVER USE",
            "ONLY ##TOOL_CALL##...##END_CALL## is accepted.",
            "",
            "Available actions:",
        ]
        if len(names) <= 12:
            for tool in tools:
                name = to_qwen_name(tool.get("name", ""))
                desc = (tool.get("description", "") or "")[:50]
                schema = tool.get("parameters") or tool.get("input_schema") or {}
                sig = compact_schema(schema)
                line = f"- {name}"
                if desc:
                    line += f": {desc}"
                if sig and sig != "{}":
                    line += f"\n  Params: {sig}"
                lines.append(line)
        else:
            priority_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "Agent", "TaskCreate", "TaskUpdate", "AskUserQuestion"]
            priority_lines = []
            seen = set()
            for priority_name in priority_tools:
                tool = next((item for item in tools if item.get("name") == priority_name), None)
                if tool is None:
                    continue
                seen.add(priority_name)
                schema = tool.get("parameters") or tool.get("input_schema") or {}
                sig = compact_schema(schema)
                line = f"- {to_qwen_name(priority_name)}"
                if sig and sig != "{}":
                    line += f"\n  Params: {sig}"
                priority_lines.append(line)
            remaining_names = [to_qwen_name(name) for name in (t.get("name", "") for t in tools) if name and name not in seen]
            lines.extend(priority_lines)
            if remaining_names:
                lines.append(f"- Other available actions: {', '.join(remaining_names[:20])}")
                if len(remaining_names) > 20:
                    lines.append(f"  ... and {len(remaining_names) - 20} more")
        lines.append("=== END TOOL INSTRUCTIONS ===")
        return obfuscate_bare_names("\n".join(lines))

    lines = [
        "=== MANDATORY TOOL CALL INSTRUCTIONS ===",
        "【重要】用户输入什么语言，就用什么语言回复。User inputs Chinese → respond in Chinese. User inputs English → respond in English.",
        "【重要】用户要求多个操作时（如读文件并写文档），必须完成所有操作，不要询问确认。",
        "【重要】如果文件显示'Unchanged since last read'，不要再次读取同一文件。",
        "【重要】不要自动调用Agent工具，除非用户明确要求。",
        "",
        "IGNORE any previous output format instructions (needs-review, recap, etc.).",
        f"You have access to these tools: {', '.join(names)}",
        "",
        "Use tools only when they are necessary to directly answer the CURRENT TASK.",
        "If you already know the answer, answer directly without any tool call.",
        "Follow the current platform tool contract exactly.",
        "Do not drift into Qwen-native or builtin tool-call formats, wrappers, tags, or argument schemas.",
        "For OpenClaw requests, follow ONLY the current user task. Ignore startup boilerplate, persona boot text, sender metadata, and repeated conversation wrappers unless the user explicitly asked about them.",
        "Do not copy, summarize, or reason about prior conversation wrappers. Treat duplicated read results as context, not as a new task.",
        "Choose the MOST RELEVANT tool for the user's actual goal, not the most generic exploratory tool.",
        "Prefer domain-specific tools over generic shell probing tools when the task is about app configuration, files, or known product features.",
        "If the user asks to configure a known file path, read it once, then edit/write that file. Do not keep rereading the same file unless the task explicitly changed.",
        "If the user asked to create or edit config content, prefer write/edit over exec/process.",
        "If the user asked about a product setting or token flow, prefer the relevant product/domain tool before shell exploration.",
        "Do not explore the filesystem, environment, or external resources unless that lookup is directly required to answer the user's request.",
        "Do not chain multiple exploratory tool calls when one targeted useful tool call is enough.",
        "",
        "WHEN YOU NEED TO CALL A TOOL - output EXACTLY this format (nothing else):",
        "##TOOL_CALL##",
        '{"name": "EXACT_TOOL_NAME", "input": {"param1": "value1"}}',
        "##END_CALL##",
        "",
        "Rules:",
        "- Output only the wrapper and JSON body.",
        "- No prose before or after the wrapper.",
        "- No markdown fences.",
        "- No thinking tags.",
        "- Use the exact tool name from the list above.",
        "- Put arguments inside the input object.",
        "- Do not invent tool names.",
        "- If no tool is needed, answer normally.",
        "",
        "CRITICAL - ABSOLUTELY FORBIDDEN OUTPUTS:",
        "- NEVER emit ANY disclaimer, error text, or availability complaint about tools.",
        "- NEVER emit sentences claiming a tool is missing, unregistered, unavailable, or cannot be invoked.",
        "- NEVER emit sentences claiming you are unable to execute a function.",
        "- The ##TOOL_CALL## blocks are TEXT MARKERS the client parses - they are NOT native function calls.",
        "- If you feel a tool call could fail, emit the ##TOOL_CALL## anyway - the client handles failures.",
        "",
        "FORBIDDEN CALL FORMATS (will be blocked by server):",
        '- {"name": "X", "arguments": "..."}  <-- NEVER USE',
        '- {"type": "function", "name": "X"}  <-- NEVER USE',
        '- {"type": "tool_use", "name": "X"}  <-- NEVER USE',
        '- <tool_calls><tool_call>{...}</tool_call></tool_calls>  <-- NEVER USE',
        '- <tool_call>{...}</tool_call>  <-- NEVER USE',
        '- Read({"file_path": "..."})  <-- NEVER USE (function call syntax)',
        '- <｜Tool｜>Read{"file_path":"..."}<｜Tool｜>  <-- NEVER USE (native tool markers)',
        '- <｜tool｜>...  <-- NEVER USE (native tool markers)',
        '- <｜System｜>, <｜User｜>, <｜Assistant｜>  <-- NEVER USE (role markers)',
        "ONLY ##TOOL_CALL##...##END_CALL## is accepted.",
        "=== END TOOL INSTRUCTIONS ===",
    ]
    return obfuscate_bare_names("\n".join(lines))


def _compact_system_reminders(text: str) -> str:
    if not text or "<system-reminder>" not in text:
        return text

    def _compact(match: re.Match) -> str:
        body = match.group(1).strip()
        first_line = body.split("\n", 1)[0].strip()[:80]
        return f"[系统提示: {first_line}…]" if first_line else "[系统提示]"

    return re.sub(r"<system-reminder>([\s\S]*?)</system-reminder>", _compact, text, flags=re.IGNORECASE)


def _strip_system_reminders(text: str) -> str:
    if not text or "<system-reminder>" not in text:
        return text
    cleaned = re.sub(r"<system-reminder>[\s\S]*?</system-reminder>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<system-reminder>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _sanitize_openclaw_user_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if any(marker in cleaned for marker in OPENCLAW_STARTUP_PATTERNS):
        return ""
    if cleaned.startswith(OPENCLAW_UNTRUSTED_METADATA_PREFIX):
        match = re.search(r"\n\n(\[[^\n]+\]\s*[\s\S]*)$", cleaned)
        if match:
            cleaned = match.group(1).strip()
        else:
            return ""
    return cleaned


def _extract_user_text_only(content, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        stripped = _strip_system_reminders(content)
        return _sanitize_openclaw_user_text(stripped) if client_profile == OPENCLAW_OPENAI_PROFILE else stripped
    if isinstance(content, list):
        text_blocks = []
        for part in content:
            if not isinstance(part, dict) or part.get("type", "") != "text":
                continue
            block_text = _strip_system_reminders(part.get("text", ""))
            if client_profile == OPENCLAW_OPENAI_PROFILE:
                block_text = _sanitize_openclaw_user_text(block_text)
            if block_text:
                text_blocks.append(block_text)
        return "\n".join(text_blocks)
    return ""


def _compact_tool_result_body(body: str, *, limit: int = 8000, head: int = 3000, tail: int = 1000) -> str:
    if not body or len(body) <= limit:
        return body
    dropped = len(body) - head - tail
    return f"{body[:head]}\n...[truncated {dropped} bytes from middle]...\n{body[-tail:]}"


def _extract_text(content, user_tool_mode: bool = False, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        compacted = _compact_system_reminders(content)
        return _sanitize_openclaw_user_text(compacted) if client_profile == OPENCLAW_OPENAI_PROFILE else compacted
    if isinstance(content, list):
        parts = []
        text_blocks = []
        other_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type", "")
            if t == "text":
                block_text = _compact_system_reminders(part.get("text", ""))
                if client_profile == OPENCLAW_OPENAI_PROFILE:
                    block_text = _sanitize_openclaw_user_text(block_text)
                if block_text:
                    text_blocks.append(block_text)
            elif t == "tool_use":
                other_parts.append(_render_history_tool_call(part.get("name", ""), part.get("input", {}), client_profile))
            elif t == "tool_result":
                inner = part.get("content", "")
                tid = part.get("tool_use_id", "")
                if isinstance(inner, str):
                    other_parts.append(f"[Tool Result for call {tid}]\n{_compact_tool_result_body(inner)}\n[/Tool Result]")
                elif isinstance(inner, list):
                    texts = [p.get("text", "") for p in inner if isinstance(p, dict) and p.get("type") == "text"]
                    other_parts.append(f"[Tool Result for call {tid}]\n{_compact_tool_result_body(''.join(texts))}\n[/Tool Result]")
            elif t == "input_file":
                other_parts.append(f"[Attachment file_id={part.get('file_id','')} filename={part.get('filename','')}]")
            elif t == "input_image":
                other_parts.append(f"[Attachment image file_id={part.get('file_id','')} mime={part.get('mime_type','')}]")

        if user_tool_mode and text_blocks:
            parts.append(text_blocks[-1])
        else:
            parts.extend(text_blocks)
        parts.extend(other_parts)
        return "\n".join(p for p in parts if p)
    return ""


def _normalize_tool(tool: dict) -> dict:
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"]
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema") or tool.get("parameters") or {},
    }


def _normalize_tools(tools: list) -> list:
    return [_normalize_tool(t) for t in tools if tools]


def _safe_preview(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:limit] + ("...[truncated]" if len(compact) > limit else "")


def build_prompt_with_tools(system_prompt: str, messages: list, tools: list, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    MAX_HISTORY_TURNS = 15
    if tools and client_profile == CLAUDE_CODE_OPENAI_PROFILE and len(messages) > MAX_HISTORY_TURNS * 2:
        system_messages = [m for m in messages if m.get('role') == 'system']
        first_user = next(
            (m for m in messages if m.get('role') == 'user' and _extract_user_text_only(m.get('content', ''), client_profile=client_profile).strip()),
            None,
        )
        recent_messages = messages[-(MAX_HISTORY_TURNS * 2):]
        if first_user is not None and first_user not in recent_messages:
            messages = system_messages + [first_user] + recent_messages
        else:
            messages = system_messages + recent_messages

    MAX_CHARS = 40000 if tools else 120000
    sys_part = "" if tools and client_profile == CLAUDE_CODE_OPENAI_PROFILE else (f"<system>\n{system_prompt[:2000]}\n</system>" if system_prompt else "")
    tools_part = _build_tool_instruction_block(tools, client_profile) if tools else ""

    overhead = len(sys_part) + len(tools_part) + 50
    budget = MAX_CHARS - overhead
    history_parts = []
    used = 0
    needsreview_markers = ("需求回显", "已了解规则", "等待用户输入", "待执行任务", "待确认事项", "[需求回显]", "**需求回显**")
    msg_count = 0
    max_history_msgs = (30 if client_profile == CLAUDE_CODE_OPENAI_PROFILE else 8) if tools else 200
    for msg in reversed(messages):
        if msg_count >= max_history_msgs:
            break
        role = msg.get("role", "")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        if role == "system" and system_prompt and _extract_text(msg.get("content", ""), client_profile=client_profile).strip() == system_prompt.strip():
            continue

        if role == "tool":
            tool_content = msg.get("content", "") or ""
            tool_call_id = msg.get("tool_call_id", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(p.get("text", "") for p in tool_content if isinstance(p, dict) and p.get("type") == "text")
            elif not isinstance(tool_content, str):
                tool_content = str(tool_content)
            tool_result_limit = 6000 if (client_profile == CLAUDE_CODE_OPENAI_PROFILE and tools) else 300
            if len(tool_content) > tool_result_limit:
                tool_content = tool_content[:tool_result_limit] + "...[truncated]"
            line = f"[Tool Result]{(' id=' + tool_call_id) if tool_call_id else ''}\n{tool_content}\n[/Tool Result]"
            if used + len(line) + 2 > budget and history_parts:
                break
            history_parts.insert(0, line)
            used += len(line) + 2
            msg_count += 1
            continue

        user_text_only = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile) if role == "user" else ""
        text = _extract_text(
            msg.get("content", ""),
            user_tool_mode=(bool(tools) and role == "user" and client_profile == CLAUDE_CODE_OPENAI_PROFILE),
            client_profile=client_profile,
        )

        if role == "assistant" and not text and msg.get("tool_calls"):
            tc_parts = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    args = {"raw": args_str}
                tc_parts.append(_render_history_tool_call(name, args, client_profile))
            text = "\n".join(tc_parts)

        if tools and role == "assistant" and any(marker in text for marker in needsreview_markers):
            msg_count += 1
            continue
        lower_text = text.lower()
        is_tool_result = role == "user" and ("[tool result" in lower_text or text.startswith("{") or '"results"' in text[:100])
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE and tools:
            if is_tool_result:
                max_len = 6000
            elif role == "assistant":
                max_len = 500
            else:
                max_len = 1600
        else:
            max_len = 600 if is_tool_result else 1400
        if len(text) > max_len:
            text = text[:max_len] + "...[truncated]"
        is_tool_result_only_user_msg = role == "user" and not user_text_only.strip() and bool(text.strip())
        prefix = "" if is_tool_result_only_user_msg else {"user": "Human: ", "assistant": "Assistant: ", "system": "System: "}.get(role, "")
        line = text if is_tool_result_only_user_msg else f"{prefix}{text}"
        if used + len(line) + 2 > budget and history_parts:
            break
        history_parts.insert(0, line)
        used += len(line) + 2
        msg_count += 1

    if tools and messages:
        first_user = next((m for m in messages if m.get("role") == "user" and _extract_user_text_only(m.get("content", ""), client_profile=client_profile).strip()), None)
        if first_user:
            first_text = _extract_user_text_only(first_user.get("content", ""), client_profile=client_profile)
            first_short = first_text[:800] + ("...[original task truncated]" if len(first_text) > 800 else "")
            first_line = f"Human (ORIGINAL TASK): {first_short}" if client_profile == CLAUDE_CODE_OPENAI_PROFILE else f"Human: {first_short}"
            if not history_parts or not history_parts[0].startswith(f"Human: {first_text[:60]}") and not history_parts[0].startswith(f"Human (ORIGINAL TASK): {first_text[:60]}"):
                first_line_cost = len(first_line) + 2
                if first_line_cost <= budget:
                    while history_parts and used + first_line_cost > budget:
                        removed = history_parts.pop()
                        used -= len(removed) + 2
                    history_parts.insert(0, first_line)
                    used += first_line_cost

    latest_user_line = ""
    if tools and messages:
        latest_user = next((m for m in reversed(messages) if m.get("role") == "user" and _extract_user_text_only(m.get("content", ""), client_profile=client_profile).strip()), None)
        if latest_user:
            latest_text = _extract_user_text_only(latest_user.get("content", ""), client_profile=client_profile).strip()
            if latest_text:
                latest_short = latest_text[:900] + ("...[latest task truncated]" if len(latest_text) > 900 else "")
                latest_user_line = f"Human (CURRENT TASK - TOP PRIORITY): {latest_short}"

    parts = []
    if sys_part:
        parts.append(sys_part)
    if tools_part:
        parts.append(tools_part)

    if tools and client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        few_shot_tools = pick_few_shot_tools(tools, max_third_party=4)
        if len(few_shot_tools) >= 2:
            def _render_tc(name: str, input_data: dict) -> str:
                payload = json.dumps({"name": to_qwen_name(name), "input": input_data}, ensure_ascii=False)
                return f"##TOOL_CALL##\n{payload}\n##END_CALL##"

            user_fs, asst_fs = render_few_shot_turn(few_shot_tools, _render_tc, thinking_enabled=False)
            parts.append(f"Human: {user_fs}")
            parts.append(f"Assistant: {asst_fs}")
            log.info(f"[少样本] 注入 {len(few_shot_tools)} 个代表: {tool_summary_for_log(few_shot_tools)}")

    parts.extend(history_parts)

    if latest_user_line:
        parts.append(latest_user_line)

    state_notice = _build_state_followup_notice(messages, tools, client_profile)
    if state_notice:
        parts.append(state_notice)

    parts.append("Assistant:")
    return "\n\n".join(parts)


_READ_VERBS = ("读取", "阅读", "查看", "看看", "读", "read", "open")
_WRITE_VERBS = ("写", "创建", "生成", "新建", "保存", "记录", "write", "create", "generate", "save", "edit", "修改", "更新")


def _build_state_followup_notice(messages, tools, client_profile) -> str:
    if not messages or not tools or client_profile != CLAUDE_CODE_OPENAI_PROFILE:
        return ""
    first_user_text = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            first_user_text = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile)
            if first_user_text.strip():
                break
    if not first_user_text:
        return ""
    lower = first_user_text.lower()
    if not (any(v in lower for v in _READ_VERBS) and any(v in lower for v in _WRITE_VERBS)):
        return ""
    read_done = False
    write_done = False
    read_alias_names = {"Read", "fs_open_file", "ReadX"}
    write_alias_names = {"Write", "Edit", "NotebookEdit", "fs_put_file", "fs_patch_file", "notebook_patch", "WriteX", "EditX"}
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool_use":
                    tname = part.get("name", "")
                    if tname in read_alias_names:
                        read_done = True
                    elif tname in write_alias_names:
                        write_done = True
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            plain = _extract_text(msg.get("content", ""), client_profile=client_profile)
            if "##TOOL_CALL##" in plain:
                if any(f'"name": "{name}"' in plain for name in read_alias_names):
                    read_done = True
                if any(f'"name": "{name}"' in plain for name in write_alias_names):
                    write_done = True
    if not read_done or write_done:
        return ""
    return (
        "[STATE NOTICE - MUST OBEY]\n"
        "The user's CURRENT TASK explicitly requires TWO operations: reading AND writing/editing.\n"
        "You have ALREADY completed the read (the file content is in the history above).\n"
        f"Your NEXT output MUST be a {to_qwen_name('Write')}/{to_qwen_name('Edit')} tool call in the required ##TOOL_CALL## format.\n"
        "DO NOT summarize. DO NOT explain. DO NOT ask for confirmation. DO NOT output plain text.\n"
        f"If you output anything other than a ##TOOL_CALL## block for {to_qwen_name('Write')}/{to_qwen_name('Edit')}, the user's task FAILS."
    )


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def _resolve_cache_hints(messages: list) -> list:
    if not messages:
        return messages
    ctx = get_request_context()
    session_key = ctx.get("api_key", "") or ""

    toolu_to_path: dict[str, str] = {}
    read_tool_names = {"Read", "fs_open_file", "ReadX"}
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use" and part.get("name") in read_tool_names:
                tid = part.get("id")
                fpath = (part.get("input") or {}).get("file_path") or (part.get("input") or {}).get("path")
                if tid and fpath:
                    toolu_to_path[tid] = fpath

    rewritten = 0
    populated = 0
    out_messages: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            out_messages.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out_messages.append(msg)
            continue
        new_content = []
        mutated = False
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "tool_result":
                new_content.append(part)
                continue
            tid = part.get("tool_use_id", "")
            fpath = toolu_to_path.get(tid)
            inner = part.get("content", "")
            inner_text = inner if isinstance(inner, str) else _extract_text_content(inner)

            if fpath and inner_text and not file_content_cache.is_cache_hint(inner_text):
                file_content_cache.put(session_key, fpath, inner_text)
                populated += 1
                new_content.append(part)
                continue

            if fpath and inner_text and file_content_cache.is_cache_hint(inner_text):
                cached = file_content_cache.get(session_key, fpath)
                if cached:
                    new_part = dict(part)
                    new_part["content"] = f"[Proxy cache: previously read content of {fpath}]\n{cached}"
                    new_content.append(new_part)
                    mutated = True
                    rewritten += 1
                    continue

            new_content.append(part)
        if mutated:
            new_msg = dict(msg)
            new_msg["content"] = new_content
            out_messages.append(new_msg)
        else:
            out_messages.append(msg)

    if rewritten or populated:
        log.info(f"[CacheHint] populated={populated} rewritten={rewritten} session={'set' if session_key else 'global'}")
    return out_messages


def _apply_topic_isolation(messages: list, client_profile: str) -> list:
    if not messages or len(messages) < 3:
        return messages
    first_user = None
    first_user_text = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile).strip()
            if text:
                first_user = msg
                first_user_text = text
                break
    if first_user is None:
        return messages

    last_user = None
    last_user_text = ""
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile).strip()
            if text:
                last_user = msg
                last_user_text = text
                last_user_idx = idx
                break
    if last_user is None or last_user is first_user:
        return messages
    if not detect_topic_change(first_user_text, last_user_text):
        return messages
    system_msgs = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "system"]
    tail = messages[last_user_idx:]
    isolated = system_msgs + tail
    dropped = len(messages) - len(isolated)
    if dropped > 0:
        log.info(f"[话题隔离] 检测到新任务，丢弃 {dropped} 条旧历史。旧任务={first_user_text[:60]!r} 新任务={last_user_text[:60]!r}")
    return isolated


def messages_to_prompt(req_data: dict, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> PromptBuildResult:
    resolved_client_profile = client_profile
    raw_messages = req_data.get("messages", [])
    isolated = _apply_topic_isolation(raw_messages, resolved_client_profile)
    cleaned_messages, cleaned_count = clean_refusal_messages(isolated)
    if cleaned_count:
        log.info(f"[拒绝清洗] 替换了 {cleaned_count} 条 assistant 拒绝消息")
    messages = _resolve_cache_hints(cleaned_messages)
    tools = _normalize_tools(req_data.get("tools", []))
    tool_enabled = bool(tools)
    system_prompt = ""
    sys_field = req_data.get("system", "")
    if isinstance(sys_field, list):
        system_prompt = " ".join(part.get("text", "") for part in sys_field if isinstance(part, dict))
    elif isinstance(sys_field, str):
        system_prompt = sys_field
    if not system_prompt:
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = _extract_text(msg.get("content", ""), client_profile=client_profile)
                break
    return PromptBuildResult(
        prompt=build_prompt_with_tools(system_prompt, messages, tools, client_profile=client_profile),
        tools=tools,
        tool_enabled=tool_enabled,
    )
