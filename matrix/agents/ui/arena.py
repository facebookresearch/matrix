"""
Sokrates Speech LLM Arena — Streamlit Prototype Demo
Demonstrates: topic/model/format selection, streaming LLM output,
3-agent debate trajectory, human-in-the-loop, and voice input.
"""

import base64
import json
import random
import time
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sokrates — Speech LLM Arena",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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
    .agent-alpha { border-color: #4f8ef7; background: #1a2744; }
    .agent-beta  { border-color: #f74f8e; background: #441a2e; }
    .agent-gamma { border-color: #4ff7a0; background: #1a4432; }
    .agent-human { border-color: #f7c94f; background: #443a1a; }
    .agent-judge { border-color: #c94ff7; background: #3a1a44; }
    .agent-name  { font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .score-box {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
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
    .format-pill {
        display: inline-block;
        background: #2a2a3a;
        color: #aaa;
        font-size: 0.8rem;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        border: 1px solid #444;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ccc;
        border-bottom: 1px solid #333;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    .iq-bar { background: #4f8ef7; height: 8px; border-radius: 4px; }
    .eq-bar { background: #f74f8e; height: 8px; border-radius: 4px; }
    .joint-bar { background: #c94ff7; height: 8px; border-radius: 4px; }
    .rollout-proponent { border-color: #4f8ef7; background: #1a2744; }
    .rollout-opponent  { border-color: #f74f8e; background: #441a2e; }
    .rollout-seed      { border-color: #888; background: #2a2a2a; }
    .topic-box {
        background: #1e1e2e; border: 1px solid #444; border-radius: 8px;
        padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 1.05rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "debate_running" not in st.session_state:
    st.session_state.debate_running = False
if "debate_finished" not in st.session_state:
    st.session_state.debate_finished = False
if "scores" not in st.session_state:
    st.session_state.scores = {
        "Alpha": {"iq": 50, "eq": 50},
        "Beta": {"iq": 50, "eq": 50},
        "Gamma": {"iq": 50, "eq": 50},
    }
if "human_turn" not in st.session_state:
    st.session_state.human_turn = False
if "awaiting_human" not in st.session_state:
    st.session_state.awaiting_human = False


# ── OpenAI client (lazy — only needed for the Arena tab) ─────────────────────
def _get_openai_client():
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI()
    return st.session_state.openai_client


# ── Debate formats ────────────────────────────────────────────────────────────
FORMATS = {
    "Quick Round-Robin (12 turns, ~12 min)": {
        "turns": 12,
        "label": "T1",
        "color": "#4f8ef7",
    },
    "Oxford Modified (18 turns, ~22 min)": {
        "turns": 18,
        "label": "T2",
        "color": "#f7a04f",
    },
    "Extended Cross-Exam (30 turns, ~38 min)": {
        "turns": 30,
        "label": "T3",
        "color": "#f74f8e",
    },
}

AGENT_COLORS = {
    "Alpha": ("#4f8ef7", "agent-alpha"),
    "Beta": ("#f74f8e", "agent-beta"),
    "Gamma": ("#4ff7a0", "agent-gamma"),
    "Human": ("#f7c94f", "agent-human"),
    "Judge": ("#c94ff7", "agent-judge"),
}

MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gemini-2.5-flash",
    "[Mock] Qwen2-Audio (offline)",
    "[Mock] LLaMA-Omni (offline)",
]

SAMPLE_TOPICS = [
    "AI systems should be granted legal personhood",
    "Universal basic income is necessary in an AI economy",
    "Social media does more harm than good to democracy",
    "Nuclear energy is essential for climate goals",
    "Humans should colonize Mars within 50 years",
]

# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Arena Configuration")

    st.markdown(
        '<div class="section-header">Debate Topic</div>', unsafe_allow_html=True
    )
    topic_choice = st.selectbox(
        "Choose a sample topic or enter custom:", ["Custom..."] + SAMPLE_TOPICS
    )
    if topic_choice == "Custom...":
        topic = st.text_input(
            "Enter your debate topic:",
            placeholder="e.g. AI should replace human judges",
        )
    else:
        topic = topic_choice
    st.session_state.topic = topic

    st.markdown(
        '<div class="section-header">Debate Format</div>', unsafe_allow_html=True
    )
    format_choice = st.selectbox("Select format:", list(FORMATS.keys()))
    fmt = FORMATS[format_choice]
    st.markdown(
        f"""
    <span class="format-pill">⏱ {fmt['label']}</span>
    <span class="format-pill">🔄 {fmt['turns']} turns</span>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">Agent Models</div>', unsafe_allow_html=True
    )
    model_alpha = st.selectbox("🔵 Alpha (PRO):", MODELS, index=0)
    model_beta = st.selectbox("🔴 Beta (CON):", MODELS, index=1)
    model_gamma = st.selectbox("🟢 Gamma (NEUTRAL):", MODELS, index=2)

    st.markdown(
        '<div class="section-header">Human-in-the-Loop</div>', unsafe_allow_html=True
    )
    human_enabled = st.toggle("Enable human participant", value=False)
    if human_enabled:
        human_position = st.radio(
            "Your position:", ["PRO", "CON", "NEUTRAL", "MODERATOR"]
        )
        human_turn_interval = st.slider("Human speaks every N turns:", 2, 6, 3)

    st.markdown(
        '<div class="section-header">Scoring Weights</div>', unsafe_allow_html=True
    )
    iq_weight = st.slider("IQ weight (what was said):", 0.0, 1.0, 0.5, 0.05)
    eq_weight = round(1.0 - iq_weight, 2)
    st.caption(f"EQ weight (how it was said): {eq_weight}")

    st.divider()
    col_start, col_reset = st.columns(2)
    with col_start:
        start_btn = st.button(
            "▶ Start",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.debate_running or not topic,
        )
    with col_reset:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if reset_btn:
        st.session_state.debate_history = []
        st.session_state.turn_count = 0
        st.session_state.debate_running = False
        st.session_state.debate_finished = False
        st.session_state.scores = {
            "Alpha": {"iq": 50, "eq": 50},
            "Beta": {"iq": 50, "eq": 50},
            "Gamma": {"iq": 50, "eq": 50},
        }
        st.session_state.awaiting_human = False
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🎙️ Sokrates Speech LLM Arena</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Three-dimensional evaluation via multi-agent debate · IQ · EQ · Joint Score</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_arena, tab_scores, tab_voice, tab_rollouts, tab_about = st.tabs(
    ["🏛️ Arena", "📊 Scores", "🎤 Voice Input", "🎧 Rollouts", "ℹ️ About"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: ARENA
# ─────────────────────────────────────────────────────────────────────────────
with tab_arena:

    # Status bar
    total_turns = fmt["turns"]
    progress_val = st.session_state.turn_count / total_turns if total_turns > 0 else 0  # type: ignore[operator]
    col_status1, col_status2, col_status3 = st.columns([3, 1, 1])
    with col_status1:
        st.progress(
            progress_val, text=f"Turn {st.session_state.turn_count} / {total_turns}"
        )
    with col_status2:
        status_icon = (
            "🟢 Running"
            if st.session_state.debate_running
            else ("✅ Finished" if st.session_state.debate_finished else "⏸ Idle")
        )
        st.metric("Status", status_icon)
    with col_status3:
        st.metric("Format", fmt["label"])

    st.divider()

    # Agent columns header
    col_a, col_b, col_g = st.columns(3)
    with col_a:
        st.markdown(
            f"<div style='color:#4f8ef7;font-weight:700;font-size:1.1rem;'>🔵 Alpha — PRO</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {model_alpha}")
    with col_b:
        st.markdown(
            f"<div style='color:#f74f8e;font-weight:700;font-size:1.1rem;'>🔴 Beta — CON</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {model_beta}")
    with col_g:
        st.markdown(
            f"<div style='color:#4ff7a0;font-weight:700;font-size:1.1rem;'>🟢 Gamma — NEUTRAL</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {model_gamma}")

    st.divider()

    # Debate trajectory display
    debate_container = st.container()
    with debate_container:
        for entry in st.session_state.debate_history:
            agent = entry["agent"]
            color, css_class = AGENT_COLORS.get(agent, ("#aaa", "agent-alpha"))
            turn_label = f"Turn {entry['turn']}" if "turn" in entry else ""
            st.markdown(
                f"""
            <div class="agent-card {css_class}">
                <div class="turn-badge">{turn_label}</div>
                <div class="agent-name" style="color:{color};">{agent}</div>
                {entry['text']}
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ── Debate engine ─────────────────────────────────────────────────────────
    AGENT_ROLES = {
        "Alpha": ("PRO", model_alpha),
        "Beta": ("CON", model_beta),
        "Gamma": ("NEUTRAL / CHALLENGER", model_gamma),
    }

    def get_system_prompt(agent_name, position, topic, history_text):
        return f"""You are {agent_name}, a debater taking the {position} position on the topic:
"{topic}"

You are in a structured 3-way debate with two other AI agents. Keep your response to 3-4 sentences maximum.
Be direct, persuasive, and use rhetorical emphasis. Reference the previous arguments if relevant.

Previous debate so far:
{history_text}

Now deliver your next argument."""

    def stream_agent_response(agent_name, position, model, topic, history):
        history_text = "\n".join([f"{e['agent']}: {e['text']}" for e in history[-6:]])
        system_prompt = get_system_prompt(agent_name, position, topic, history_text)

        # Use mock for offline models
        if model.startswith("[Mock]"):
            mock_responses = [
                f"As the {position} voice, I argue that {topic.lower()} is fundamentally about human values and long-term consequences.",
                f"The evidence clearly supports the {position} position: we must consider both immediate impacts and systemic effects.",
                f"My opponents overlook the critical nuance here — the {position} perspective accounts for what others ignore.",
                f"To summarize the {position} case: the data, the ethics, and the logic all converge on this conclusion.",
            ]
            text = random.choice(mock_responses)
            # Simulate streaming
            for word in text.split():
                yield word + " "
                time.sleep(0.04)
        else:
            try:
                stream = _get_openai_client().chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}],
                    max_tokens=150,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                yield f"[API error: {e}]"

    def update_scores(agent_name):
        """Simulate score updates after each turn."""
        delta_iq = random.randint(-3, 8)
        delta_eq = random.randint(-2, 6)
        st.session_state.scores[agent_name]["iq"] = min(
            100, max(0, st.session_state.scores[agent_name]["iq"] + delta_iq)
        )
        st.session_state.scores[agent_name]["eq"] = min(
            100, max(0, st.session_state.scores[agent_name]["eq"] + delta_eq)
        )

    # ── Run debate ────────────────────────────────────────────────────────────
    if start_btn and topic:
        st.session_state.debate_running = True
        st.session_state.debate_finished = False
        agent_order = ["Alpha", "Beta", "Gamma"]

        for turn in range(1, total_turns + 1):  # type: ignore[operator]
            st.session_state.turn_count = turn
            agent_name = agent_order[(turn - 1) % 3]
            position, model = AGENT_ROLES[agent_name]
            color, css_class = AGENT_COLORS[agent_name]

            # Human-in-the-loop check
            if human_enabled and turn % human_turn_interval == 0:
                st.session_state.awaiting_human = True
                st.session_state.debate_history.append(
                    {
                        "agent": "Human",
                        "turn": turn,
                        "text": "⏳ <em>Waiting for human input...</em> (see Voice Input tab)",
                    }
                )
                st.rerun()
                break

            # Stream agent response
            with debate_container:
                st.markdown(
                    f"""
                <div class="agent-card {css_class}">
                    <div class="turn-badge">Turn {turn}</div>
                    <div class="agent-name" style="color:{color};">{agent_name} — {position}</div>
                """,
                    unsafe_allow_html=True,
                )

                placeholder = st.empty()
                full_text = ""
                for chunk in stream_agent_response(
                    agent_name, position, model, topic, st.session_state.debate_history
                ):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)

                st.markdown("</div>", unsafe_allow_html=True)

            st.session_state.debate_history.append(
                {"agent": agent_name, "turn": turn, "text": full_text}
            )
            update_scores(agent_name)

            # Small pause between turns
            time.sleep(0.3)

        # Judge verdict
        if not st.session_state.awaiting_human:
            with debate_container:
                color, css_class = AGENT_COLORS["Judge"]
                verdict_placeholder = st.empty()
                verdict_text = ""
                verdict_prompt = f"""You are the Judge in a 3-way debate on: "{topic}"
Here is the full debate:
{chr(10).join([f"{e['agent']}: {e['text']}" for e in st.session_state.debate_history])}

Provide a 3-sentence verdict: who made the strongest IQ argument, who had the best rhetorical delivery (EQ), and who wins overall. Be concise."""

                try:
                    stream = _get_openai_client().chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[{"role": "user", "content": verdict_prompt}],
                        max_tokens=200,
                        stream=True,
                    )
                    st.markdown(
                        f"""
                    <div class="agent-card {css_class}">
                        <div class="turn-badge">Final Verdict</div>
                        <div class="agent-name" style="color:{color};">⚖️ Judge</div>
                    """,
                        unsafe_allow_html=True,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            verdict_text += chunk.choices[0].delta.content
                            verdict_placeholder.markdown(verdict_text + "▌")
                    verdict_placeholder.markdown(verdict_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    verdict_text = f"[Judge error: {e}]"
                    verdict_placeholder.markdown(verdict_text)

                st.session_state.debate_history.append(
                    {"agent": "Judge", "turn": total_turns + 1, "text": verdict_text}  # type: ignore[operator]
                )

            st.session_state.debate_running = False
            st.session_state.debate_finished = True
            st.success("✅ Debate complete! See the Scores tab for full results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SCORES
# ─────────────────────────────────────────────────────────────────────────────
with tab_scores:
    st.markdown("### 📊 Live Scoring Dashboard")
    st.caption(
        "Scores update after each turn. IQ = argument quality (LLM judge on transcript). EQ = prosodic delivery (simulated)."
    )

    if not st.session_state.debate_history:
        st.info("Start a debate to see live scores.")
    else:
        scores = st.session_state.scores
        col1, col2, col3 = st.columns(3)

        for col, (agent, color) in zip(
            [col1, col2, col3],
            [("Alpha", "#4f8ef7"), ("Beta", "#f74f8e"), ("Gamma", "#4ff7a0")],
        ):
            with col:
                iq = scores[agent]["iq"]
                eq = scores[agent]["eq"]
                joint = round(iq * iq_weight + eq * eq_weight)
                st.markdown(
                    f"<div style='color:{color};font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;'>{agent}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"**IQ Score** (what was said)")
                st.progress(iq / 100)
                st.markdown(
                    f"<div style='font-size:1.8rem;font-weight:700;color:{color};'>{iq}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"**EQ Score** (how it was said)")
                st.progress(eq / 100)
                st.markdown(
                    f"<div style='font-size:1.8rem;font-weight:700;color:{color};'>{eq}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"**Joint Score** (IQ–EQ alignment)")
                st.progress(joint / 100)
                st.markdown(
                    f"<div style='font-size:2rem;font-weight:700;color:{color};'>{joint} 🏆</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("### Turn-by-Turn Transcript")
        for entry in st.session_state.debate_history:
            agent = entry["agent"]
            color, _ = AGENT_COLORS.get(agent, ("#aaa", ""))
            st.markdown(
                f"**Turn {entry.get('turn', '?')} — <span style='color:{color}'>{agent}</span>:** {entry['text']}",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: VOICE INPUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_voice:
    st.markdown("### 🎤 Voice Input — Human-in-the-Loop")
    st.markdown(
        """
    When human-in-the-loop mode is enabled and it is your turn, record your argument here.
    Your audio will be transcribed via Whisper and injected into the debate as your turn.
    """
    )

    col_voice1, col_voice2 = st.columns([1, 1])

    with col_voice1:
        st.markdown("#### Record your argument")
        audio_input = st.audio_input("🎙️ Click to record", sample_rate=16000)

        if audio_input is not None:
            st.audio(audio_input)
            st.success("Audio recorded! Transcribing...")

            with st.spinner("Transcribing with Whisper..."):
                try:
                    transcript = _get_openai_client().audio.transcriptions.create(
                        model="whisper-1",
                        file=("audio.wav", audio_input, "audio/wav"),
                    )
                    transcribed_text = transcript.text
                    st.session_state.human_transcript = transcribed_text
                    st.success(f"✅ Transcribed: *{transcribed_text}*")
                except Exception as e:
                    # Fallback for demo if Whisper not available
                    transcribed_text = "[Whisper transcription unavailable in this environment — text input used instead]"
                    st.session_state.human_transcript = transcribed_text
                    st.warning(f"Whisper not available: {e}")

    with col_voice2:
        st.markdown("#### Or type your argument")
        text_input = st.text_area(
            "Type your debate argument:",
            height=120,
            placeholder="Enter your argument here if voice is not available...",
        )
        if text_input:
            st.session_state.human_transcript = text_input

        if st.session_state.get("human_transcript"):
            st.markdown(f"**Your argument:** {st.session_state.human_transcript}")

        if st.button(
            "💬 Submit to Debate",
            type="primary",
            disabled=not st.session_state.get("human_transcript"),
        ):
            if st.session_state.awaiting_human:
                # Replace the waiting placeholder with actual human input
                for entry in st.session_state.debate_history:
                    if entry["agent"] == "Human" and "Waiting" in entry["text"]:
                        entry["text"] = st.session_state.human_transcript
                        break
                st.session_state.awaiting_human = False
                st.session_state.human_transcript = ""
                st.success(
                    "✅ Your argument has been added to the debate! Return to the Arena tab."
                )
                st.rerun()
            else:
                st.info(
                    "No debate is currently waiting for human input. Start a debate with Human-in-the-Loop enabled."
                )

    st.divider()
    st.markdown("#### 🔊 Listen to Agent Outputs (TTS Simulation)")
    st.caption(
        "In the full Sokrates system, each agent's output would be synthesized via TTS and played back here with prosodic analysis."
    )

    if st.session_state.debate_history:
        agent_filter = st.selectbox(
            "Select agent to replay:", ["All"] + ["Alpha", "Beta", "Gamma", "Judge"]
        )
        entries = [
            e
            for e in st.session_state.debate_history
            if agent_filter == "All" or e["agent"] == agent_filter
        ]
        for entry in entries[-3:]:  # Show last 3
            agent = entry["agent"]
            color, _ = AGENT_COLORS.get(agent, ("#aaa", ""))
            st.markdown(
                f"**<span style='color:{color}'>{agent}</span> (Turn {entry.get('turn','?')}):** {entry['text']}",
                unsafe_allow_html=True,
            )
            # TTS placeholder
            if st.button(
                f"▶ Play {agent} Turn {entry.get('turn','?')}",
                key=f"tts_{agent}_{entry.get('turn','?')}",
            ):
                st.info(
                    "🔊 TTS playback would stream audio here via Pipecat/ElevenLabs in the full system."
                )
    else:
        st.info("Run a debate first to replay agent outputs.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ROLLOUT VIEWER
# ─────────────────────────────────────────────────────────────────────────────
ROLLOUT_AGENT_STYLE = {
    "proponent": ("#4f8ef7", "rollout-proponent", "Proponent (FOR)"),
    "opponent": ("#f74f8e", "rollout-opponent", "Opponent (AGAINST)"),
}


def _load_rollout_results(path: str) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _load_yaml_config(path: str) -> dict | None:
    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _decode_audio(audio_entry: dict) -> bytes | None:
    data = audio_entry.get("data")
    if data:
        try:
            return base64.b64decode(data)
        except Exception:
            return None
    return None


def _get_rollout_topic(game: dict) -> str:
    task = game.get("task", {})
    if task is None:
        return "unknown"
    return task.get("topic", task.get("question", str(task)))


with tab_rollouts:
    st.markdown("### 🎧 Rollout Viewer")
    st.caption("Browse past simulation results and listen to agent audio outputs.")

    col_file, col_cfg = st.columns(2)
    with col_file:
        rollout_path = st.text_input(
            "Results JSONL file:",
            value="omni_debate_results.jsonl",
            key="rollout_path",
        )
    with col_cfg:
        rollout_config_path = st.text_input(
            "Config YAML (optional):",
            value="",
            key="rollout_config_path",
        )

    load_rollouts_btn = st.button("📂 Load Results", key="load_rollouts")

    if load_rollouts_btn:
        if not Path(rollout_path).exists():
            st.error(f"File not found: {rollout_path}")
        else:
            st.session_state.rollout_games = _load_rollout_results(rollout_path)
            st.session_state.rollout_file = rollout_path
            if rollout_config_path and Path(rollout_config_path).exists():
                st.session_state.rollout_config = _load_yaml_config(rollout_config_path)
            else:
                st.session_state.rollout_config = None

    if "rollout_games" in st.session_state and st.session_state.rollout_games:
        games = st.session_state.rollout_games
        rollout_config = st.session_state.get("rollout_config")

        st.markdown(
            f"**{len(games)}** rollout(s) loaded from `{st.session_state.get('rollout_file', '')}`"
        )

        if rollout_config:
            with st.expander("Run config", expanded=False):
                st.json(rollout_config)

        # Game selector
        def _game_label(i):
            g = games[i]
            t = _get_rollout_topic(g)
            short = t[:70] + "…" if len(t) > 70 else t
            gid = g.get("id", i)
            return f"#{gid}  {short}"

        selected_game_idx = st.selectbox(
            "Select a rollout:",
            range(len(games)),
            format_func=_game_label,
            key="rollout_game_select",
        )

        # Summary stats
        successes = sum(1 for g in games if g.get("status", {}).get("success"))
        errors = sum(1 for g in games if g.get("status", {}).get("error"))
        st.caption(
            f"✅ {successes} success · ❌ {errors} error · 📊 {len(games)} total"
        )

        st.divider()

        # Selected game
        game = games[selected_game_idx]
        r_topic = _get_rollout_topic(game)
        r_history = game.get("history", [])
        r_status = game.get("status", {})

        status_emoji = (
            "✅"
            if r_status.get("success")
            else ("❌" if r_status.get("error") else "⏹")
        )
        st.markdown(
            f'<div class="topic-box">{status_emoji} <strong>Topic:</strong> {r_topic}</div>',
            unsafe_allow_html=True,
        )

        col_m1, col_m2, col_m3 = st.columns(3)
        debate_entries = [h for h in r_history if h["agent"] in ROLLOUT_AGENT_STYLE]
        with col_m1:
            st.metric("Debate Turns", len(debate_entries))
        with col_m2:
            st.metric("Seed", game.get("seed", "—"))
        with col_m3:
            st.metric("Trial", game.get("trial", "—"))

        st.divider()

        # Render turns
        for turn_idx, entry in enumerate(r_history):
            agent = entry.get("agent", "unknown")
            response = entry.get("response", {})
            text = response.get("text", "")
            audio_list = response.get("audio") or []
            usage = response.get("usage", {})

            style_info = ROLLOUT_AGENT_STYLE.get(agent)
            if style_info:
                color, css_class, label = style_info
            else:
                color, css_class, label = "#888", "rollout-seed", agent.capitalize()

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

            # Audio playback
            for audio_idx, audio_entry in enumerate(audio_list):
                audio_bytes = _decode_audio(audio_entry)
                transcript = audio_entry.get("transcript", "")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    if transcript:
                        st.caption(f"📝 Transcript: {transcript}")
                else:
                    st.caption("⚠️ Audio data could not be decoded.")

        # Raw JSON
        with st.expander("Raw JSON", expanded=False):
            display_game = json.loads(json.dumps(game, default=str))
            for h in display_game.get("history", []):
                for a in h.get("response", {}).get("audio", []):
                    if "data" in a and len(str(a["data"])) > 100:
                        a["data"] = (
                            str(a["data"])[:80] + f"... ({len(str(a['data']))} chars)"
                        )
            st.json(display_game)
    elif "rollout_games" in st.session_state:
        st.info("No rollouts found in the file.")
    else:
        st.info(
            "Enter a results JSONL file path above and click **Load Results** to browse rollouts."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
### About Sokrates

**Sokrates** is a three-dimensional evaluation platform for spoken LLMs, staging structured debates between three audio-native models to assess not just *what* they say (IQ Score) but *how* they say it (EQ Score) and whether the right words are emphasized (Joint Score).

#### The Three Scores

| Dimension | Score | What it measures | Method |
|---|---|---|---|
| Content | **IQ Score** | Argument quality, logic, evidence | LLM judge on transcript |
| Delivery | **EQ Score** | Pitch, pauses, rate, emphasis | Acoustic classifier + LLM on prosodic captions |
| Alignment | **Joint Score** | IQ–EQ coherence | Algorithmic PIA + LLM judge |

#### Debate Formats

| Format | Turns | Duration | Context Tier |
|---|---|---|---|
| Quick Round-Robin | 12 | ~12 min | T1 — all models |
| Oxford Modified | 18 | ~22 min | T2 — long context |
| Extended Cross-Exam | 30 | ~38 min | T3 — 30+ min endurance |

#### This Demo

This Streamlit prototype demonstrates the full interaction surface:
- **Topic & model selection** via sidebar
- **Debate format selection** (12/18/30 turns)
- **Streaming LLM output** token-by-token via OpenAI API
- **3-agent debate trajectory** with color-coded turn display
- **Live score dashboard** updating after each turn
- **Human-in-the-loop** voice recording via `st.audio_input` + Whisper transcription
- **TTS replay** placeholder for the full speech pipeline

#### Full System Architecture

The production Sokrates system uses:
- **LangGraph** for debate state machine orchestration
- **Pipecat** for real-time audio streaming between models
- **MATRIX** for HPC-scale offline debate generation
- **Parselmouth / openSMILE** for prosodic feature extraction
    """
    )
