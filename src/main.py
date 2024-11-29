import logging
import os
import pickle

import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_bytes

import config
from src.asr import transcribe
from src.llm import (create_chunk_for_gemini, generate_response_with_genai,
                     get_relevant_content)
from src.ocr_with_pytesseract import detect_text_blocks
from src.prompts import (audio_extract_text_prompt, audio_prompt,
                         images_extract_text_prompt, images_prompt)
from src.utils import (clear_folders_content, display_result, get_youtube,
                       video_to_audio)

st.set_page_config(
    page_title="Youtube video analyser",
    page_icon=None,
    layout="wide",  # "centered" or "wide",
    initial_sidebar_state="auto",
    menu_items=None,
)



################################
###### SETTINGS ###############
################################

with st.expander("Settings"):
    with st.form("Home", clear_on_submit=True):
        model = st.selectbox("Choose OpenAI model for audio translation", ("GPT 3.5", "GPT 4"), key="_model")
        ## Add a slider to the sidebar:
        entity = st.text_input(
            "Entity",
            key="_entity",
            placeholder=config.PLACEHOLDER_ENTITY,
        )
        entity_description = st.text_area(
            "Entity Description",
            key="_entity_description",
            placeholder=config.PLACEHOLDER_ENTITY_DESCRIPTION,
        )

        # Every form must have a submit button.
        submitted = st.form_submit_button(
            "Submit",
        )
    if submitted:
        with open(config.PERSISTENCE_SESSION_STATE_PATH, "wb") as f:
            pickle.dump(st.session_state.to_dict(), f)


###################################
############## Main ###############
###################################

session_state = {}
if os.path.isfile(config.PERSISTENCE_SESSION_STATE_PATH):
    with open(config.PERSISTENCE_SESSION_STATE_PATH, "rb") as f:
        session_state = pickle.load(f)

# initilaized section variable:
for key, value in session_state.items():
    st.session_state[key[1:]] = value

############
# Side bar #
###########


if 'sidebar_youtube_link' not in st.session_state:
    st.session_state['sidebar_youtube_link'] = ''



source = "Youtube"
with st.sidebar.form("settings"):
    ## Add a selectbox to the sidebar:
    openai_key = st.text_input(
        "Add your OpenAI key",
        key="openai_key",
        type="password",
        placeholder=config.PLACEHOLDER_OPEN_AI_APIKEY,
    )
    gemini_key = st.text_input(
        "Add your Google AI Studio api_key",
        key="gemini_key",
        type="password",
        placeholder=config.PLACEHOLDER_GEMINI_APIKEY,
    )

    sidebar_youtube_link = st.text_input(
        "Youtube Link", key="youtube_link", placeholder=config.PLACEHOLDER_LINK
    )
    st.session_state["sidebar_youtube_link"] = sidebar_youtube_link
    st.session_state["video_link"] = sidebar_youtube_link

    uploaded_video_file = st.file_uploader("Choose a local video file", type=["mp4", "avi"])
    uploaded_audio_file = st.file_uploader("Choose a local audio file", type=["mp3", "wav"])
    uploaded_images_files = st.file_uploader("Choose multiple local images files ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    uploaded_pdf_file = st.file_uploader("Choose a local PDF file", type=["pdf"])


    # initialize variables
    st.session_state['uploaded_audio'] = ""
    st.session_state["sidebar_local_video_file"] = ""
    st.session_state["sidebar_local_image"]=[]
    st.session_state['uploaded_pdf_images']=[]

    # processing of the uploaded video file
    if uploaded_video_file:
        
        file_extension = uploaded_video_file.name.split(".")[-1].lower()
        if file_extension in ["mp4", "avi"]:
            video_bytes = uploaded_video_file.getvalue()
            with open(config.VIDEO_FILE, "wb") as f:
                f.write(video_bytes)
            st.session_state["sidebar_local_video_file"] = config.VIDEO_FILE
            
        else:
            st.error("Unsupported file format. Please choose a supported file format.")


    # processing audio file 
    if uploaded_audio_file:
        

        # Enregistrer le fichier audio localement
        audio_save_path = os.path.join(config.AUDIO_FILE, uploaded_audio_file.name)
        with open(audio_save_path, "wb") as f:
            f.write(uploaded_audio_file.getbuffer())
        st.session_state['uploaded_audio'] = audio_save_path

       
    # processing of the uploaded images file
    if uploaded_images_files:
        
        for image_file in uploaded_images_files:
            image_bytes = image_file.getvalue()
            image_extension = image_file.name.split(".")[-1].lower()
            if image_extension in ["jpg", "jpeg", "png"]:
                # Traitement pour les images
                image_file_path = os.path.join(config.IMAGE_FILE, image_file.name)
                with open(image_file_path, "wb") as f:
                    f.write(image_bytes)
                st.session_state["sidebar_local_image"].append(image_file_path)
                
            else:
                st.error("Unsupported file format. Please choose a supported file format.")

    


  
    # processing of the uploaded pdf file
    if uploaded_pdf_file:
        

        # Enregistrer le fichier localement
        save_pdf_path = os.path.join(config.IMAGE_FROM_PDF_FILE , uploaded_pdf_file.name)
        with open(save_pdf_path, "wb") as f:
            f.write(uploaded_pdf_file.getbuffer())
        images_folder = os.path.dirname(save_pdf_path)
        images = convert_from_bytes(uploaded_pdf_file.read())

        for i, image in enumerate(images):
            image_from_pdf_path = os.path.join(images_folder, f"{uploaded_pdf_file.name}_page_{i+1}.png")
            image.save(image_from_pdf_path, "PNG")
            st.session_state['uploaded_pdf_images'].append(image_from_pdf_path)

       
    # Every form must have a submit button.
    sidebar_send = st.form_submit_button("Send")



#### initialize variables
youtube_relevant_text_temp=""
audio_relevant_text_temp =""
images_relevant_text_temp =""
pdf_images_relevant_text_temp=""
video_uploaded_relevant_text_temp=""
relevant_content_video_uploaded_text=""

if sidebar_send:
    
    
    result_images_analysis_and_summary = ""
    result_pdf_analysis_and_summary = ""
    result_audio_analysis_and_summary = ""
    result_youtube_analysis_and_summary = ""
    result_video_analysis_and_summary = ""
    
    client_openai = OpenAI(api_key=st.session_state.openai_key)
    
    ### processing for extract text from the youtube video and pass it to genai
    if st.session_state.get("sidebar_youtube_link"):
        
        audio_path, video_title = get_youtube(st.session_state.sidebar_youtube_link )
        logging.info("ASR_video")
        transcription = transcribe(audio_file_path=audio_path, client=client_openai, lang="fr")
        
        
        chunks = create_chunk_for_gemini(
            text=transcription.strip(), api_key=st.session_state.gemini_key, chunk_overlap=10, separator="\n\n"
        )
        # extract relevent content
        for chunk in chunks:
            prompt_for_extracting_relevant_text_for_youtube  =  audio_extract_text_prompt.format(
                        entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                        entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                        audio_transcription = chunk )
            youtube_relevant_text_temp += generate_response_with_genai(prompt_for_extracting_relevant_text_for_youtube , st.session_state.gemini_key )
        
                      
        relevant_content_youtube_text = get_relevant_content(
            text_extract=youtube_relevant_text_temp,
            relevant_content=transcription,
            window_size=3,
            separator="\n\n",
            )
        
       
        youtube_final_prompt = audio_prompt.format(entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                    entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    audio_transcription = relevant_content_youtube_text)
        result_youtube_analysis_and_summary =  generate_response_with_genai(youtube_final_prompt, st.session_state.gemini_key )
        
            
        
         
        
    ###### processing of the audio uploaded file 
    if st.session_state.get("uploaded_audio"):
            
            
            logging.info("ASR_audio_uploaded_file")
            transcription = transcribe(audio_file_path=st.session_state['uploaded_audio'], client=client_openai, lang="fr")
        
            chunks = create_chunk_for_gemini(
                text=transcription.strip(), api_key=st.session_state.gemini_key, chunk_overlap=10, separator="\n\n"
            )
            for chunk in chunks:
                prompt_for_extracting_relevant_text_for_audio  =  audio_extract_text_prompt.format(
                        entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                        entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                        audio_transcription = chunk)
                
                audio_relevant_text_temp += generate_response_with_genai(prompt_for_extracting_relevant_text_for_audio , st.session_state.gemini_key )
            
            relevant_content_audio_text = get_relevant_content(
                text_extract=audio_relevant_text_temp,
                relevant_content=transcription,
                window_size=3,
                separator="\n\n",
                )
            


            audio_final_prompt = audio_prompt.format(entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                    entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    audio_transcription =  relevant_content_audio_text)
            result_audio_analysis_and_summary =  generate_response_with_genai(audio_final_prompt, st.session_state.gemini_key )
            #print("2222222222222222222222222222222:",result_audio_analysis_and_summary )
          
            
            
    ############### processing uploaded images file
    
    if st.session_state["sidebar_local_image"]:
        transcription =""
        for i in st.session_state["sidebar_local_image"]:
            transcription +=  detect_text_blocks(i)
        
        #print(transcription )
        logging.info("images processing")
        
        chunks = create_chunk_for_gemini(
            text = transcription.strip(), api_key = st.session_state.gemini_key, separator="\n\n"
        )

        for chunk in chunks:
                prompt_for_extracting_relevant_text_for_image  =  images_extract_text_prompt.format(
                        entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                        entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                        images_transcription = chunk)
                images_relevant_text_temp += generate_response_with_genai(prompt_for_extracting_relevant_text_for_image , st.session_state.gemini_key )
                
    

        images_final_prompt = images_prompt.format(entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                    entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    extracted_text_images = images_relevant_text_temp )
        
        result_images_analysis_and_summary =  generate_response_with_genai(images_final_prompt, st.session_state.gemini_key )
        #print("33333333333333333333333333333 :",result_images_analysis_and_summary)


        
    ######################" processing pdf file"
    if st.session_state['uploaded_pdf_images']:
        transcription =""
        for i in st.session_state["uploaded_pdf_images"]:
            transcription +=  detect_text_blocks(i)
            logging.info("pdf processing")
        
        chunks = create_chunk_for_gemini(
                text=transcription.strip(), api_key= st.session_state.gemini_key, chunk_overlap=10, separator="\n\n"
            )
        for chunk in chunks:
            prompt_for_extracting_relevant_text_for_pdf_image  =  images_extract_text_prompt.format(
                        entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                        entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                        images_transcription = chunk)
            pdf_images_relevant_text_temp += generate_response_with_genai(prompt_for_extracting_relevant_text_for_pdf_image , st.session_state.gemini_key )
        pdf_final_prompt = images_prompt.format(entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                    entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    extracted_text_images = pdf_images_relevant_text_temp )
        result_pdf_analysis_and_summary =  generate_response_with_genai(pdf_final_prompt, st.session_state.gemini_key )
        #print("444444444444444444444444444444 : ",result_pdf_analysis_and_summary)
            


    ########processing of uploaded video
    if st.session_state["sidebar_local_video_file"]:
        audio_path = video_to_audio(st.session_state.sidebar_local_video_file,config.AUDIO_FILE_FROM_VIDEO)
        # ASR
        logging.info("ASR uploaded video")
        transcription = transcribe(audio_file_path=audio_path, client=client_openai, lang="fr")
        chunks = create_chunk_for_gemini(
            text=transcription.strip(), api_key=st.session_state.gemini_key, chunk_overlap=10, separator="\n\n"
        )
        for chunk in chunks:
            prompt_for_extracting_relevant_text_for_video_uploaded  =  audio_extract_text_prompt.format(
                    entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                    entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    audio_transcription = chunk)
            video_uploaded_relevant_text_temp += generate_response_with_genai(prompt_for_extracting_relevant_text_for_video_uploaded  , st.session_state.gemini_key )
            
        relevant_content_video_uploaded_text = get_relevant_content(
            text_extract=video_uploaded_relevant_text_temp,
            relevant_content=transcription,
            window_size=1,
            separator="\n\n",
            )
        
        video_final_prompt = audio_prompt.format(entity=st.session_state.get("entity") or config.PLACEHOLDER_ENTITY,
                    entity_description=st.session_state.get("entity_description")or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    audio_transcription = relevant_content_video_uploaded_text )
        result_video_analysis_and_summary =  generate_response_with_genai(video_final_prompt, st.session_state.gemini_key )
        
        



    # Check which result variables exist
    result_variables = [result_images_analysis_and_summary,
                        result_pdf_analysis_and_summary,
                        result_audio_analysis_and_summary,
                        result_youtube_analysis_and_summary,
                        result_video_analysis_and_summary]
  
    st.chat_message("assistant").write(f"{'hello'.center(80)}")

    
        # Affichage des résultats pour chaque type
    if result_images_analysis_and_summary:
        display_result("Images", "Source", result_images_analysis_and_summary)
    if result_pdf_analysis_and_summary:
        display_result("PDF", "Source", result_pdf_analysis_and_summary)
    if result_audio_analysis_and_summary:
        display_result("Audio", "Source", result_audio_analysis_and_summary)
    if result_youtube_analysis_and_summary:
        display_result("YouTube", "Source", result_youtube_analysis_and_summary)
    if result_video_analysis_and_summary:
        display_result("Video", "Source", result_video_analysis_and_summary)

    # Liste des dossiers à vider
    folders_to_clear = [
        config.AUDIO_FILE ,
        config.IMAGE_FROM_PDF_FILE ,
        config.IMAGE_FILE,
        config.YOUTUBE_FILES_PATH,
        config.VIDEO_FILES_PATH]
    clear_folders_content(folders_to_clear)
    























if __name__ == "__main__":
    if os.path.isfile(config.PERSISTENCE_SESSION_STATE_PATH):
        os.remove(config.PERSISTENCE_SESSION_STATE_PATH)
