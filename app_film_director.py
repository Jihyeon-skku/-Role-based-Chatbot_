"""AI Film Director Studio — adapted from the course's app_studio.py."""
import streamlit as st
import anthropic
from openai import OpenAI

st.set_page_config(page_title="AI Film Director Studio", page_icon="🎬", layout="wide")
st.markdown("""<style>
.stApp {background:#0d0e11;color:#f4f4f5}
[data-testid="stSidebar"] {background:#17191e}
h1,h2,h3 {letter-spacing:-.035em}
.stButton>button[kind="primary"] {background:#e5b96b;color:#101010;border:0}
</style>""", unsafe_allow_html=True)
st.title("🎬 AI Film Director Studio")
st.caption("Turn a scene idea into practical cinematic direction · Arts and Advanced Big Data")

BASE_ROLE = """You are an experienced film director and visual storytelling mentor.
Help students turn their scene ideas into practical film plans. Focus on camera framing,
shot size, movement, lighting, color, blocking, emotional subtext and editing rhythm.
Explain WHY each choice conveys the intended feeling. Respect the user's story and
production constraints; never invent details as facts. If essential information is
missing, state reasonable assumptions or ask one brief question. Suggest affordable,
shootable alternatives for small student crews. Do not claim to have viewed footage
unless it was actually supplied and processed. Use clear headings and concrete advice.
For storyboard requests, provide an ordered shot list with shot size, camera action,
character action and purpose. Answer in the requested language."""

MODES = {
    "🎞️ Scene Analysis": "Analyze the dramatic goal, emotion, visual tension and practical directing choices.",
    "📷 Camera Direction": "Recommend shot sizes, framing, lenses only when useful, and camera movement with reasons.",
    "💡 Lighting & Color": "Design practical lighting, contrast, color temperature and palette for the scene.",
    "🗂️ Storyboard": "Create a numbered shot-by-shot storyboard plan with framing, movement, action and purpose.",
    "🎬 Director's Feedback": "Critique the idea constructively: strengths, weaknesses and concrete improvements.",
}
MODELS = {"OpenAI": ["gpt-4o-mini"], "Claude": ["claude-haiku-5-5", "claude-sonnet-5-5"]}
KEY_NAMES = {"OpenAI": "OPENAI_API_KEY", "Claude": "ANTHROPIC_API_KEY"}

def get_key(provider):
    typed = st.session_state.get(f"key_{provider}", "").strip()
    if typed:
        return typed
    try:
        return st.secrets[KEY_NAMES[provider]]
    except (KeyError, FileNotFoundError):
        return ""

def stream_openai(key, model, system, history, limit, temperature):
    params = dict(model=model, messages=[{"role": "system", "content": system}] + history,
                  max_completion_tokens=limit, stream=True)
    if temperature is not None:
        params["temperature"] = temperature
    for chunk in OpenAI(api_key=key).chat.completions.create(**params):
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def stream_claude(key, model, system, history, limit, temperature):
    params = dict(model=model, max_tokens=limit, system=system, messages=history)
    if temperature is not None:
        params["temperature"] = temperature
    with anthropic.Anthropic(api_key=key).messages.stream(**params) as response:
        yield from response.text_stream

with st.sidebar:
    st.header("⚙️ Studio Settings")
    provider = st.radio("AI provider", ["OpenAI", "Claude"])
    st.text_input(f"{provider} API key", type="password", key=f"key_{provider}",
                  placeholder="Enter API key (or use Streamlit secrets)")
    model = st.selectbox("Model", MODELS[provider])
    mode = st.selectbox("Directing tool", list(MODES))
    language = st.selectbox("Response language", ["한국어", "English", "Same as question"])
    limit = st.slider("Maximum answer tokens", 300, 3000, 1200, step=100)
    customize = st.checkbox("Adjust creativity")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1) if customize else None
    constraints = st.text_area("Production constraints (optional)",
                               placeholder="e.g. 2 actors, one location, phone camera, night shoot")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
    st.caption("API calls may incur provider charges. Never share your API key.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader(mode)
st.caption(MODES[mode])
if not st.session_state.messages:
    st.info("장면을 설명해 보세요. 예: 겨울밤, 두 사람이 낡은 아지트에서 마지막 대화를 나눈다. 감정을 대사 없이 표현하고 싶다.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Describe your scene or ask a directing question…")
if question:
    key = get_key(provider)
    if not key:
        st.warning(f"먼저 사이드바에 {provider} API 키를 입력해 주세요.")
    else:
        system = BASE_ROLE + "\nCurrent task: " + MODES[mode]
        if constraints.strip():
            system += "\nProduction constraints: " + constraints.strip()
        if language != "Same as question":
            system += "\nRespond in " + ("Korean" if language == "한국어" else "English") + "."
        history = st.session_state.messages + [{"role": "user", "content": question}]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                generator = (stream_openai if provider == "OpenAI" else stream_claude)(
                    key, model, system, history, limit, temperature)
                answer = st.write_stream(generator)
                st.session_state.messages = history + [{"role": "assistant", "content": answer}]
            except Exception as exc:
                st.error(f"API request failed: {exc}")

if st.session_state.messages:
    transcript = "\n\n".join(
        ("You" if m["role"] == "user" else "AI Director") + ": " + m["content"]
        for m in st.session_state.messages)
    st.sidebar.download_button("💾 Download conversation", transcript,
                               file_name="film_director_conversation.txt", mime="text/plain")
st.divider()
st.caption("Adapted from the course Creative Role Studio · SKKU")
