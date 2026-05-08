"""
Trajectory Viewer — Streamlit app for browsing multi-agent rollout results.
Supports text-only trajectories and trajectories with inline audio playback.
Loads only one page at a time to avoid blocking on large files.
"""

import base64
import json
from pathlib import Path

import streamlit as st

PAGE_SIZE = 20

st.set_page_config(
    page_title="Trajectory Viewer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .agent-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .rollout-proponent { border-color: #4f8ef7; background: #1a2744; }
    .rollout-opponent  { border-color: #f74f8e; background: #441a2e; }
    .rollout-default   { border-color: #888; background: #2a2a2a; }
    .agent-name {
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .turn-badge {
        display: inline-block;
        background: #333;
        color: #aaa;
        font-size: 0.75rem;
        padding: 0.1rem 0.5rem;
        border-radius: 20px;
        margin-bottom: 0.4rem;
    }
    .topic-box {
        background: #1e1e2e; border: 1px solid #444; border-radius: 8px;
        padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 1.05rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

AGENT_STYLES = {
    "proponent": ("#4f8ef7", "rollout-proponent", "Proponent (FOR)"),
    "opponent": ("#f74f8e", "rollout-opponent", "Opponent (AGAINST)"),
}


def _build_line_index(path: str) -> list[int]:
    """Scan file and return byte offsets for each non-empty line."""
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


def _load_page(path: str, offsets: list[int], start: int, count: int) -> list[dict]:
    """Read and parse `count` lines starting at index `start`."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for i in range(start, min(start + count, len(offsets))):
            f.seek(offsets[i])
            line = f.readline()
            if line.strip():
                results.append(json.loads(line))
    return results


def _decode_audio(audio_entry: dict) -> bytes | None:
    data = audio_entry.get("data")
    if data:
        try:
            return base64.b64decode(data)
        except Exception:
            return None
    return None


def _get_topic(game: dict) -> str:
    task = game.get("task", {})
    if task is None:
        return "unknown"
    return task.get("topic", task.get("question", str(task)))


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">📜 Trajectory Viewer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Browse multi-agent rollout trajectories · text and audio</div>',
    unsafe_allow_html=True,
)

# ── Sidebar: file selection & pagination ─────────────────────────────────────
with st.sidebar:
    st.markdown("## Load Trajectories")

    rollout_path = st.text_input(
        "Results JSONL file:",
        value="",
        key="rollout_path",
        placeholder="path/to/results.jsonl",
    )

    load_btn = st.button("Load", type="primary", use_container_width=True)

    if load_btn:
        if not rollout_path or not Path(rollout_path).exists():
            st.error(f"File not found: {rollout_path}")
        else:
            st.session_state.line_offsets = _build_line_index(rollout_path)
            st.session_state.rollout_file = rollout_path
            st.session_state.page = 0

    if "line_offsets" in st.session_state and st.session_state.line_offsets:
        total = len(st.session_state.line_offsets)
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        page = st.session_state.get("page", 0)

        st.divider()
        st.caption(f"{total} trajectories")

        col_prev, col_info, col_next = st.columns([1, 1, 1])
        with col_prev:
            if st.button("Prev", use_container_width=True, disabled=page <= 0):
                st.session_state.page = page - 1
                st.session_state.pop("rollout_game_select", None)
                st.rerun()
        with col_info:
            st.markdown(
                f"<div style='text-align:center;padding:0.4rem 0;font-size:0.9rem;'>"
                f"Page {page + 1} / {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            if st.button(
                "Next", use_container_width=True, disabled=page >= total_pages - 1
            ):
                st.session_state.page = page + 1
                st.session_state.pop("rollout_game_select", None)
                st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────
if "line_offsets" not in st.session_state or not st.session_state.line_offsets:
    st.info("Enter a results JSONL path in the sidebar and click **Load**.")
    st.stop()

offsets = st.session_state.line_offsets
rollout_file = st.session_state.rollout_file
page = st.session_state.get("page", 0)
page_start = page * PAGE_SIZE

games = _load_page(rollout_file, offsets, page_start, PAGE_SIZE)

if not games:
    st.info("No trajectories on this page.")
    st.stop()


def _game_label(i):
    g = games[i]
    t = _get_topic(g)
    short = t[:70] + "…" if len(t) > 70 else t
    global_idx = page_start + i
    gid = g.get("id", global_idx)
    return f"#{gid}  {short}"


selected_idx = st.selectbox(
    "Select trajectory:",
    range(len(games)),
    format_func=_game_label,
    key="rollout_game_select",
)

game = games[selected_idx]
topic = _get_topic(game)
history = game.get("history", [])
status = game.get("status", {})

status_emoji = "✅" if status.get("success") else ("❌" if status.get("error") else "⏹")
st.markdown(
    f'<div class="topic-box">{status_emoji} <strong>Topic:</strong> {topic}</div>',
    unsafe_allow_html=True,
)

st.metric("Turns", len(history))

st.divider()

# ── Render turns ─────────────────────────────────────────────────────────────
for turn_idx, entry in enumerate(history):
    agent = entry.get("agent", "unknown")
    response = entry.get("response", {})
    text = response.get("text", "")
    audio_list = response.get("audio") or []
    usage = response.get("usage", {})

    style_info = AGENT_STYLES.get(agent)
    if style_info:
        color, css_class, label = style_info
    else:
        color, css_class, label = "#888", "rollout-default", agent.capitalize()

    tokens = usage.get("completion_tokens", "")
    tokens_str = f" · {tokens} tok" if tokens else ""
    audio_badge = " · 🔊" if audio_list else ""

    st.markdown(
        f"""<div class="agent-card {css_class}">
            <div class="turn-badge">Turn {turn_idx}{tokens_str}{audio_badge}</div>
            <div class="agent-name" style="color:{color};">{label}</div>
            {text}
        </div>""",
        unsafe_allow_html=True,
    )

    for audio_entry in audio_list:
        audio_bytes = _decode_audio(audio_entry)
        transcript = audio_entry.get("transcript", "")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if transcript:
                st.caption(f"Transcript: {transcript}")
        else:
            st.caption("Audio data could not be decoded.")

# ── Raw JSON ─────────────────────────────────────────────────────────────────
with st.expander("Raw JSON", expanded=False):
    display_game = json.loads(json.dumps(game, default=str))
    for h in display_game.get("history", []):
        for a in h.get("response", {}).get("audio", []):
            if "data" in a and len(str(a["data"])) > 100:
                a["data"] = str(a["data"])[:80] + f"... ({len(str(a['data']))} chars)"
    st.json(display_game)
