
import os
import pickle
import time

import streamlit as st
from openai import OpenAI

import config
from src.utils import (clear_folders_content, display_result,
                       process_audio_analysis_and_summary,
                       process_images_analysis_and_summary,
                       process_pdf_analysis_and_summary,
                       process_uploaded_video_analysis_and_summary,
                       process_youtube_analysis_and_summary)

st.set_page_config(
    page_title="News analyser",
    page_icon="🤖",
    layout="wide",  # "centered" or "wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

################################
###### SETTINGS ###############
################################

with st.expander("Settings"):
    with st.form("Home", clear_on_submit=True):
        # model = st.selectbox(
        #     "Choose OpenAI model for audio translation",
        #     ("GPT 3.5",
        #      "GPT 4"),
        #     key="_model")
        # Add a slider to the sidebar:
        entity = st.text_input(
            "Entité",
            key="_entity",
            placeholder=config.PLACEHOLDER_ENTITY,
        )
        entity_description = st.text_area(
            "Description de l'entité",
            key="_entity_description",
            placeholder=config.PLACEHOLDER_ENTITY_DESCRIPTION,
        )

        # Every form must have a submit button.
        submitted = st.form_submit_button(
            "Envoyer",
        )
    if submitted:
        with open(config.PERSISTENCE_SESSION_STATE_PATH, "wb") as f:
            pickle.dump(st.session_state.to_dict(), f)


###################################
############## Main ###############
###################################

# Load session state from a file if it exists, and transfer the data to
# Streamlit's session state
session_state = {}
if os.path.isfile(config.PERSISTENCE_SESSION_STATE_PATH):
    with open(config.PERSISTENCE_SESSION_STATE_PATH, "rb") as f:
        session_state = pickle.load(f)


for key, value in session_state.items():
    st.session_state[key[1:]] = value


############
# Side bar #
###########


if 'sidebar_youtube_link' not in st.session_state:
    st.session_state['sidebar_youtube_link'] = ''


source = "Youtube"
with st.sidebar.form("settings"):
    # Add a selectbox to the sidebar:
    # openai_key = st.text_input(
    #     "Add your OpenAI key",
    #     key="openai_key",
    #     type="password",
    #     placeholder=config.PLACEHOLDER_OPEN_AI_APIKEY,
    # )
    # # gemini_key = st.text_input(
    #     "Add your Google AI Studio api_key",
    #     key="gemini_key",
    #     type="password",
    #     placeholder=config.PLACEHOLDER_GEMINI_APIKEY,
    # )

    sidebar_youtube_link = st.text_input(
        "Lien youtube", key="youtube_link", placeholder=config.PLACEHOLDER_LINK
    )
    st.session_state["sidebar_youtube_link"] = sidebar_youtube_link
    # st.session_state["video_link"] = sidebar_youtube_link

    uploaded_video_file = st.file_uploader(
        "Choisir un fichier vidéo local", type=["mp4", "avi"])
    uploaded_audio_file = st.file_uploader(
        "Choisir un fichier audio local", type=["mp3", "wav"], key="uploaded_video_file")
    uploaded_images_files = st.file_uploader(
        "Choisir plusieurs fichiers d'images locaux ", type=[
            "jpg", "jpeg", "png"], accept_multiple_files=True)
    uploaded_pdf_files = st.file_uploader(
        "Choisir un fichier pdf local", type=["pdf"], accept_multiple_files=True)

    # Every form must have a submit button.
    sidebar_send = st.form_submit_button("Envoyer")


if sidebar_send:
    start_time = time.time()
    #  initialize the final variables that contain the results to display
    result_images_analysis_and_summary = ""
    result_pdf_analysis_and_summary = ""
    result_audio_analysis_and_summary = ""
    result_youtube_analysis_and_summary = ""
    result_video_analysis_and_summary = ""

    client_openai = OpenAI(api_key=config.OPENAI_API_KEY)

    # Process the uploaded YouTube video link, transcribe, and extract

    if st.session_state.get("sidebar_youtube_link"):
        result_youtube_analysis_and_summary = process_youtube_analysis_and_summary(
            youtube_link=st.session_state.get("sidebar_youtube_link"),
            gemini_key=config.GEMINI_API_KEY,
            client_openai=client_openai,
            entity_description=st.session_state.get("entity"),
            entity=st.session_state.get("entity_description")
        )

    # Process the uploaded audio file, transcribe, and extract relevant content
    if uploaded_audio_file:
        result_audio_analysis_and_summary = process_audio_analysis_and_summary(
            uploaded_audio_file=uploaded_audio_file,
            gemini_key=config.GEMINI_API_KEY,
            client_openai=client_openai,
            entity_description=st.session_state.get("entity_description"),
            entity=st.session_state.get("entity")
        )

    # Process the uploaded images files, transcribe, and extract relevant

    if uploaded_images_files:
        result_images_analysis_and_summary = process_images_analysis_and_summary(
            uploaded_images_files=uploaded_images_files,
            gemini_key=config.GEMINI_API_KEY,
            entity_description=st.session_state.get("entity_description"),
            entity=st.session_state.get("entity")
        )

    # # Process the uploaded pdfs files, transcribe, and extract relevant content

    if uploaded_pdf_files:
        result_pdf_analysis_and_summary = process_pdf_analysis_and_summary(
            uploaded_pdf_files=uploaded_pdf_files,
            gemini_key=config.GEMINI_API_KEY,
            entity_description=st.session_state.get("entity_description"),
            entity=st.session_state.get("entity")
        )

    # Process the uploaded video file, transcribe, and extract relevant content

    if uploaded_video_file:
        result_video_analysis_and_summary = process_uploaded_video_analysis_and_summary(
            uploaded_video_file=uploaded_video_file,
            client_openai=client_openai,
            gemini_key=config.GEMINI_API_KEY,
            entity_description=st.session_state.get("entity_description"),
            entity=st.session_state.get("entity")
        )

    # Check which result final  variables exist
    result_variables = [result_images_analysis_and_summary,
                        result_pdf_analysis_and_summary,
                        result_audio_analysis_and_summary,
                        result_youtube_analysis_and_summary,
                        result_video_analysis_and_summary]

    st.chat_message("assistant").write(f"{'hello'.center(80)}")

    # display the results for each final variables
    if result_images_analysis_and_summary:
        display_result("Images", "Source", result_images_analysis_and_summary)
    if result_pdf_analysis_and_summary:
        display_result("PDF", "Source", result_pdf_analysis_and_summary)
    if result_audio_analysis_and_summary:
        display_result("Audio", "Source", result_audio_analysis_and_summary)
    if result_youtube_analysis_and_summary:
        display_result(
            "YouTube",
            "Source",
            result_youtube_analysis_and_summary)
    if result_video_analysis_and_summary:
        display_result("Video", "Source", result_video_analysis_and_summary)

    #  cler the folders where we stored the data
    folders_to_clear = [
        config.AUDIO_FILE,
        config.IMAGE_FROM_PDF_FILE,
        config.IMAGE_FILE,
        config.YOUTUBE_FILES_PATH,
        config.VIDEO_FILES_PATH]
    clear_folders_content(folders_to_clear)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution total : {execution_time} secondes")


if __name__ == "__main__":
    if os.path.isfile(config.PERSISTENCE_SESSION_STATE_PATH):
        os.remove(config.PERSISTENCE_SESSION_STATE_PATH)
