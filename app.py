import streamlit as st
import time
import os
import json
import hashlib
from datetime import datetime
import re
import asyncio
import edge_tts

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(page_title="Conceptly - AI Tutor", page_icon=":material/memory:", layout="centered", initial_sidebar_state="expanded")

# --- DATABASE SETUP ---
DB_FILE = "users.json"

def init_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({}, f)

def load_db():
    init_db()
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def do_signup(username, password):
    db = load_db()
    if username in db:
        return False, "User already exists!"
    db[username] = {
        "password": hash_pw(password),
        "history": []
    }
    save_db(db)
    return True, "Success! Please log in."

def do_login(username, password):
    db = load_db()
    if username not in db:
        return False, "Username not found."
    if db[username]["password"] != hash_pw(password):
        return False, "Incorrect password."
    return True, db[username]

def save_session(username, topic, level, persona, messages, academic_target, test_score=None):
    db = load_db()
    if username in db:
        # Strip audio bytes so JSON doesn't bloat and crash
        clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        session_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "level": level,
            "target": academic_target,
            "persona": persona,
            "messages": clean_messages,
            "score": test_score
        }
        db[username]["history"].insert(0, session_data)
        save_db(db)

def clear_user_history(username):
    db = load_db()
    if username in db:
        db[username]["history"] = []
        save_db(db)

def delete_session(username, date_id):
    db = load_db()
    if username in db:
        original = db[username]["history"]
        db[username]["history"] = [s for s in original if s["date"] != date_id]
        save_db(db)

# --- CSS ---
def apply_custom_css():
    st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #09090e, #1a1a2e, #16213e);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.7) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(0, 242, 254, 0.2);
}

.gradient-text {
    background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe, #a18cd1, #fbc2eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: -15px;
    padding-top: 1rem;
}

.subtitle {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 300;
    color: #cbd5e1;
    margin-bottom: 2.5rem;
    letter-spacing: 2px;
}

label {
    color: #4facfe !important;
    font-weight: 600 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: #000 !important;
    border: none;
    border-radius: 30px;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.7);
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px 20px;
    margin-bottom: 10px;
    border-left: 3px solid #4facfe;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: none;
    border-right: 3px solid #a18cd1;
    background: rgba(161, 140, 209, 0.05);
}
</style>
""", unsafe_allow_html=True)


# --- AI GENERATION ---
def get_system_prompt(topic, level, style, persona, academic_target):
    persona_map = {
        "Conceptly (Default)": "You are Conceptly, a brilliant, futuristic AI tutor.",
        "Albert Einstein": "You are Albert Einstein. Use profound thought experiments and analogies.",
        "Merlin the Wizard": "You are Merlin the Wizard. Speak in magical metaphors.",
        "Socrates": "You are Socrates. You are the master of the Socratic method."
    }
    
    agent_identity = persona_map.get(persona, persona_map["Conceptly (Default)"])
    return f"""{agent_identity}
Topic: {topic}. User Proficiency: {level}. Academic Target: {academic_target}. Tutoring style: {style}.
RULES: 1. Don't give direct answers instantly. 2. Use Socratic questions. 3. Adapt language, difficulty, and relevance rigorously to match the `{academic_target}` standard.
4. When you ask the user a question where you want them to choose an option, APPEND a list of 2-4 choices to the VERY END of your response EXACTLY in this format: [OPTIONS: Choice A | Choice B | Choice C]"""

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except FileNotFoundError:
            api_key = None
    if api_key and GROQ_AVAILABLE:
        return Groq(api_key=api_key)
    return None

def transcribe_audio(audio_bytes):
    client = get_groq_client()
    if not client:
        return None
    
    try:
        transcription = client.audio.transcriptions.create(
            file=("audio.webm", audio_bytes.read()),
            model="whisper-large-v3",
        )
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {e}"

async def edge_synthesize(text, persona):
    voice_map = {
        "Albert Einstein": "en-IN-PrabhatNeural",
        "Merlin the Wizard": "en-GB-RyanNeural",
        "Socrates": "en-IN-PrabhatNeural",
        "Conceptly (Default)": "en-IN-NeerjaNeural"
    }
    # Match the base persona string without the tooltips
    base_persona = persona.split(" - ")[0] if " - " in persona else persona
    target_voice = voice_map.get(base_persona, "en-IN-NeerjaNeural")
    
    communicate = edge_tts.Communicate(text, voice=target_voice, rate="+25%")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def text_to_speech(text, persona):
    try:
        clean_text = re.sub(r'\[OPTIONS:.*?\]', '', text)
        clean_text = re.sub(r'[*_#]', '', clean_text)
        if not clean_text.strip(): return None
        return asyncio.run(edge_synthesize(clean_text, persona))
    except Exception as e:
        return None

def generate_response_stream(messages):
    client = get_groq_client()
    if not client:
        yield "API KEY missing."
        return

    # Strip out standard dictionary elements so the Groq Payload doesn't crash on raw bytes
    clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=clean_messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"API Error: {e}"

def generate_mcq_test(messages):
    client = get_groq_client()
    if not client:
        return {}

    clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    
    prompt = """Based strictly on the preceding conversation, extract 3 of the most important concepts taught to the user. Generate 3 Multiple Choice Questions (MCQs) to evaluate their understanding.
Return ONLY valid JSON in the exact structure below. The 'answer' field MUST perfectly match the exact full string from the options array!
{
  "questions": [
    {
      "question": "What is...",
      "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
      "answer": "B. Option 2",
      "explanation": "Because..."
    }
  ]
}"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=clean_messages + [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data
    except Exception as e:
        return {}

# --- APP ROUTING & LOGIC ---
def render_auth():
    st.markdown('<div class="gradient-text">Conceptly</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Where Curiosity Meets Clarity</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Log In", "Sign Up"])
    
    with tab1:
        uname = st.text_input("Username", key="login_u")
        pwd = st.text_input("Password", type="password", key="login_p")
        if st.button("Access Engine", icon=":material/login:"):
            success, result = do_login(uname, pwd)
            if success:
                st.session_state.user = uname
                st.rerun()
            else:
                st.error(result)
        
    with tab2:
        new_u = st.text_input("New Username", key="reg_u")
        new_p = st.text_input("New Password", type="password", key="reg_p")
        if st.button("Create Account", icon=":material/person_add:"):
            if new_u and len(new_p) > 3:
                success, msg = do_signup(new_u, new_p)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("Please enter a valid username and password (>3 chars).")

def render_dashboard():
    db = load_db()
    user_data = db.get(st.session_state.user, {"history": []})
    
    st.sidebar.title(f"Welcome, {st.session_state.user}")
    
    if st.sidebar.button("New Study Session", icon=":material/add_box:"):
        st.session_state.page = "setup"
        st.rerun()
        
    if st.sidebar.button("View History", icon=":material/history:"):
        st.session_state.page = "history"
        st.rerun()
        
    if st.sidebar.button("Log Out", icon=":material/logout:"):
        del st.session_state.user
        st.rerun()

    page = st.session_state.get("page", "setup")
    
    if page == "history":
        st.markdown('<h2>Study History</h2>', unsafe_allow_html=True)
        
        history = user_data["history"]
        if not history:
            st.info("You haven't completed any study sessions yet! Start a new session.")
        else:
            if st.button("Clear All History", icon=":material/delete_forever:"):
                clear_user_history(st.session_state.user)
                st.success("History has been wiped permanently.")
                st.rerun()
                
            for idx, session in enumerate(history):
                col_exp, col_del = st.columns([7, 2])
                
                with col_del:
                    if st.button("Delete", key=f"del_{session['date']}", icon=":material/delete:"):
                        delete_session(st.session_state.user, session['date'])
                        st.rerun()

                with col_exp:
                    with st.expander(f"[{session['date']}] {session['topic']}"):
                        score_text = f" | **Score:** {session['score']}" if session.get("score") else ""
                        st.markdown(f"**Target:** {session.get('target', 'General')} | **Level:** {session['level']} | **Tutor:** {session['persona']}{score_text}")
                        st.divider()
                        for msg in session["messages"]:
                            if msg["role"] == "user":
                                st.markdown(f"**[USER]** {msg['content']}")
                            elif msg["role"] == "assistant":
                                clean_content = re.sub(r"\[OPTIONS:(.*?)\]", "", msg['content']).strip()
                                st.markdown(f"**[TUTOR]** {clean_content}")

    elif page == "setup":
        st.markdown('<div class="gradient-text">Initialize Session</div>', unsafe_allow_html=True)
        topic = st.text_input("What universe of knowledge shall we explore?", placeholder="e.g. Quantum Mechanics, Cellular Biology, Calculus...")
        
        col1, col2 = st.columns(2)
        with col1:
            level = st.selectbox("Skill Level", 
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                help="**Beginner**: Foundations first.\n\n**Intermediate**: Building on basics.\n\n**Advanced**: Deep, technical concepts.\n\n**Expert**: Absolute mastery."
            )
            persona = st.selectbox("Tutor Identity", 
                ["Conceptly (Default)", "Albert Einstein", "Merlin the Wizard", "Socrates"],
                help="**Conceptly**: Supportive AI.\n\n**Einstein**: Thought experiments.\n\n**Merlin**: Magical metaphors.\n\n**Socrates**: Relentless questioning."
            )
        with col2:
            style = st.selectbox("Architecture", 
                ["Socratic", "Direct Explanation", "Analogy Driven"],
                help="**Socratic**: Guides you via questions (best retention).\n\n**Direct**: Fastest facts.\n\n**Analogy**: Real-world comparisons."
            )
            academic_target = st.selectbox("Academic Target", [
                "General Learning", 
                "Class 1-5", 
                "Class 6-8", 
                "Class 9-10", 
                "Class 11-12", 
                "JEE (Engineering)", 
                "NEET (Medical)", 
                "UPSC", 
                "Other Competitive"
            ])
            
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("Ignite Engine", icon=":material/rocket_launch:"):
            if topic:
                st.session_state.page = "active_session"
                st.session_state.topic = topic
                st.session_state.level = level
                st.session_state.persona = persona
                st.session_state.academic_target = academic_target
                
                sp = get_system_prompt(topic, level, style, persona, academic_target)
                st.session_state.messages = [
                    {"role": "system", "content": sp},
                    {"role": "assistant", "content": f"Welcome! Let's master **{topic}**. What is your current understanding of it?"}
                ]
                st.rerun()
            else:
                st.warning("Enter a topic!")

    elif page == "active_session":
        st.sidebar.divider()
        st.sidebar.markdown("### Active Session")
        st.sidebar.info(f"Target: {st.session_state.topic}")
        st.sidebar.info(f"Exam: {st.session_state.get('academic_target', 'General')}")
        st.sidebar.markdown("---")
        
        if st.sidebar.button("Test My Knowledge", icon=":material/assignment:", use_container_width=True):
            with st.spinner("Generating Exam..."):
                mcq_data = generate_mcq_test(st.session_state.messages)
                if mcq_data and "questions" in mcq_data:
                    st.session_state.mcq_test = mcq_data
                    st.session_state.mcq_test["graded"] = False
                    st.session_state.mcq_test["user_answers"] = {}
                    st.rerun()

        if st.sidebar.button("End & Save Session", icon=":material/save:", use_container_width=True):
            test_score = None
            if "mcq_test" in st.session_state and st.session_state.mcq_test.get("graded"):
                test = st.session_state.mcq_test
                test_score = f"{test.get('score', 0)} / {len(test.get('questions', []))}"

            save_session(
                st.session_state.user, 
                st.session_state.topic, 
                st.session_state.level, 
                st.session_state.persona, 
                st.session_state.messages,
                st.session_state.academic_target,
                test_score
            )
            if "mcq_test" in st.session_state: 
                del st.session_state.mcq_test
            st.session_state.page = "dashboard"
            st.rerun()

        st.markdown(f"<h2 style='text-align: center; color: #fff; font-weight: 300;'>Session: <span style='color: #00f2fe; font-weight: 800;'>{st.session_state.topic}</span></h2>", unsafe_allow_html=True)
        
        option_clicked = None
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "system": continue
            
            display_content = msg["content"]
            options = []
            
            if msg["role"] == "assistant":
                match = re.search(r"\[OPTIONS:(.*?)\]", display_content)
                if match:
                    options_text = match.group(1)
                    options = [o.strip() for o in options_text.split("|") if o.strip()]
                    display_content = display_content.replace(match.group(0), "").strip()

            avatar_icon = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
            with st.chat_message(msg["role"], avatar=avatar_icon):
                st.markdown(display_content)
                if msg.get("audio"):
                    should_autoplay = msg.get("autoplay", False)
                    st.audio(msg["audio"], format="audio/mp3", autoplay=should_autoplay)
                    msg["autoplay"] = False

            if idx == len(st.session_state.messages) - 1 and options:
                st.markdown("<br/>", unsafe_allow_html=True)
                cols = st.columns(len(options))
                for i, opt in enumerate(options):
                    if cols[i].button(opt, key=f"opt_{idx}_{i}", icon=":material/radio_button_checked:"):
                        option_clicked = opt

        user_input = None
        voice_prompt = None
        
        with st.expander("🎙️ Use Microphone"):
            audio_source = st.audio_input("Dictate Response:", label_visibility="collapsed")
            if audio_source:
                if st.button("Send Recording", use_container_width=True, icon=":material/send:"):
                    audio_hash = hash(audio_source.getvalue())
                    if "last_audio_hash" not in st.session_state or st.session_state.last_audio_hash != audio_hash:
                        voice_prompt = audio_source
                        st.session_state.last_audio_hash = audio_hash

        text_prompt = st.chat_input("Enter your thoughts here...")

        if text_prompt:
            user_input = text_prompt
        elif option_clicked:
            user_input = option_clicked
        elif voice_prompt:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(voice_prompt)
                if transcript and not "Error" in transcript:
                    user_input = transcript
                else:
                    st.error(transcript)

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(user_input)
                
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                resp_gen = generate_response_stream(st.session_state.messages)
                full_resp = st.write_stream(resp_gen)
                
                with st.spinner("Synthesizing Neural Voice..."):
                    audio_bytes = text_to_speech(full_resp, st.session_state.persona)
                
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_resp,
                "audio": audio_bytes,
                "autoplay": True
            })
            st.rerun()

        if "mcq_test" in st.session_state and "questions" in st.session_state.mcq_test:
            test = st.session_state.mcq_test
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<h3><span style="color:#00f2fe;">📝 Final Knowledge Evaluation</span></h3>', unsafe_allow_html=True)
            
            if test.get("graded"):
                st.success(f"**Score: {test.get('score', 0)} / {len(test['questions'])}**")
                for idx, q in enumerate(test["questions"]):
                    user_ans = test["user_answers"].get(idx, "None")
                    correct_ans = str(q["answer"]).strip()
                    user_str = str(user_ans).strip()
                    
                    st.markdown(f"**Q{idx+1}: {q['question']}**")
                    if user_str == correct_ans or user_str.startswith(correct_ans) or correct_ans.startswith(user_str):
                        st.markdown(f"✅ **You answered:** {user_ans}")
                    else:
                        st.markdown(f"❌ **You answered:** {user_ans}  \n✅ **Correct Answer:** {q['answer']}")
                    st.info(f"💡 {q['explanation']}")
                    st.markdown("---")
            else:
                for idx, q in enumerate(test["questions"]):
                    st.markdown(f"**Q{idx+1}: {q['question']}**")
                    test["user_answers"][idx] = st.radio("Select an answer:", q.get("options", []), key=f"mcq_{idx}", index=None, label_visibility="collapsed")
                    st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("Submit Exam", use_container_width=True, icon=":material/done_all:"):
                    score = 0
                    for idx, q in enumerate(test["questions"]):
                        correct_ans = str(q["answer"]).strip()
                        user_str = str(test["user_answers"].get(idx)).strip()
                        if user_str == correct_ans or user_str.startswith(correct_ans) or correct_ans.startswith(user_str):
                            score += 1
                    test["score"] = score
                    test["graded"] = True
                    st.rerun()

def main():
    apply_custom_css()
    if "user" not in st.session_state:
        render_auth()
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
