import streamlit as st
import speech_recognition as sr
import pyttsx3

def voice_chat_input(placeholder="Speak or type...", key="main", help_text=None):
    # Initialize session state variables
    if f"voice_input_{key}" not in st.session_state:
        st.session_state[f"voice_input_{key}"] = ""
    if f"voice_transcript_{key}" not in st.session_state:
        st.session_state[f"voice_transcript_{key}"] = ""
    if f"show_voice_component_{key}" not in st.session_state:
        st.session_state[f"show_voice_component_{key}"] = False

    # Voice recording button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🎤", key=f"record_button_{key}"):
            st.session_state[f"show_voice_component_{key}"] = True

    # Voice recording logic
    if st.session_state[f"show_voice_component_{key}"]:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Please speak clearly.")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        try:
            text = recognizer.recognize_google(audio)
            st.session_state[f"voice_transcript_{key}"] = text
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")
        finally:
            st.session_state[f"show_voice_component_{key}"] = False

    # If transcription exists, put it directly in text box
    if f"voice_transcript_{key}" in st.session_state and st.session_state[f"voice_transcript_{key}"]:
        st.session_state[f"voice_input_{key}"] = st.session_state[f"voice_transcript_{key}"]
        st.session_state[f"voice_transcript_{key}"] = ""  # Clear it

    # Text input box (shows typed or spoken text)
    user_input = st.text_input(
        placeholder,
        value=st.session_state[f"voice_input_{key}"],
        key=f"user_input_{key}",
        help=help_text
    )

    # Update state if user types manually
    st.session_state[f"voice_input_{key}"] = user_input
    return user_input


def speak_text(text):
    """Simple text-to-speech output"""
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.say(text)
    engine.runAndWait()
