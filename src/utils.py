import os
import shutil
import uuid

import moviepy.editor as mp
import streamlit as st
import yt_dlp
from pdf2image import convert_from_bytes

import config
from src.asr import transcribe
from src.llm import create_chunk_for_gemini, generate_response_with_genai
from src.ocr_with_pytesseract import detect_text_blocks, detect_text_blocks_pdf
from src.prompts import (audio_extract_text_prompt, audio_prompt,
                         images_extract_text_prompt, images_prompt)


def video_to_audio(video_file: str, audio_path: str) -> str:
    """
    Convert a video file to an audio file.

    Parameters:
    - video_file (str): The path to the input video file.
    - audio_path (str): The path where the output audio file will be saved.

    Returns:
    - str: The path to the converted audio file.

    Example:
    >>> input_video = "input_video.mp4"
    >>> output_audio = "output_audio.mp3"
    >>> audio_path = video_to_audio(input_video, output_audio)
    """
    # Load the video clip
    clip = mp.VideoFileClip(video_file)

    # Write the audio file
    clip.audio.write_audiofile(audio_path)

    # Return the path to the audio file
    return audio_path


def get_youtube(video_url: str) -> tuple:
    """
    Download a YouTube video and convert it to audio.

    Parameters:
    - video_url (str): The URL of the YouTube video.

    Returns:
    - tuple: A tuple containing the path to the audio file and the title of the video.

    Example:
    >>> video_url = "https://www.youtube.com/watch?v=your_video_id"
    >>> audio_file, video_title = get_youtube(video_url)
    """

    # Options for YouTube-DL downloader
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    }

    # Download the video information without downloading the video itself
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

        # Get the title of the video
        video_title = info["title"]

        # Prepare filename for video download
        abs_video_path = ydl.prepare_filename(info)

        # Download and process the video
        ydl.process_info(info)

        # Move the downloaded video to a specific directory
        shutil.move(abs_video_path, config.YOUTUBE_FILE)

    # Convert the downloaded video to audio
    audio_file = video_to_audio(
        config.YOUTUBE_FILE,
        config.AUDIO_FILE_FROM_YOUTUBE)

    # Return the path to the audio file and the title of the video
    return audio_file, video_title


def save_image(image_bytes: bytes, file_name: str, extension: str) -> str:
    """
    Save image bytes to a file.

    Parameters:
    - image_bytes (bytes): The bytes representing the image.
    - file_name (str): The name of the file to save.
    - extension (str): The extension of the image file.

    Returns:
    - str: The path to the saved image file.

    Example:
    >>> image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01...'
    >>> file_name = "image"
    >>> extension = "png"
    >>> image_path = save_image(image_bytes, file_name, extension)
    """

    # Create a temporary folder if it doesn't exist
    temp_folder = "temp_images"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Construct the path to save the image file
    image_path = os.path.join(temp_folder, f"{file_name}.{extension}")

    # Write the image bytes to the file
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Return the path to the saved image file
    return image_path


def display_result(title: str, source: str, result: str):
    """
    Display the result with title, source, and icon.

    Parameters:
    - title (str): The title of the result.
    - source (str): The source of the result.
    - result (str): The content of the result.

    Returns:
    - None

    Example:
    >>> display_result("My Title", "Images", "My image content")
    """

    # Mapping of specific icons for each source
    source_icons = {
        "Images": "📷",
        "PDF": "📄",
        "Audio": "🔊",
        "YouTube": "🎥",
        "Video": "📹"
    }

    # Displaying the title, source, and icon on the same line
    st.write(
        f"<font color='red'>{title}</font> <font color='black'>{source}</font> {source_icons.get(title, '')}",
        unsafe_allow_html=True)

    # Displaying the result
    st.write(result)


def clear_folders_content(folders):
    """
    Empty the contents of a list of folders.

    This function iterates over the provided list of folder paths and removes all files within each folder.
    It does not remove subfolders or their contents.

    Args:
        folders (list of str): A list containing the paths to the folders to be emptied.

    Raises:
        OSError: If a folder specified in the 'folders' list does not exist or cannot be emptied.

    Example:
        >>> clear_folders_content(["folder1", "folder2"])
    """

    for folder in folders:
        # Check if the folder exists
        if not os.path.exists(folder):
            raise OSError(f"Folder '{folder}' does not exist.")

        # Iterate over files in the folder and remove them
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            # Check if it's a file and remove it
            if os.path.isfile(file_path):
                os.remove(file_path)


def generate_unique_name() -> str:
    """
    Generates a unique name using UUID version 4.

    Returns:
    str: A unique name generated using UUID version 4.
    """
    # Generate a unique name using UUID version 4
    return str(uuid.uuid4())


def save_uploaded_audio_file(uploaded_audio_file, save_directory):
    """
    Saves the uploaded audio file locally and returns the path.

    Args:
        uploaded_audio_file (UploadedFile): The uploaded audio file object.
        save_directory (str): The directory where the audio file will be saved.

    Returns:
        str: The path to the saved audio file.
    """
    if uploaded_audio_file:
        # Save the audio file locally
        audio_save_path = os.path.join(
            save_directory, uploaded_audio_file.name)
        with open(audio_save_path, "wb") as f:
            f.write(uploaded_audio_file.getbuffer())
        return audio_save_path
    else:
        return None


def process_uploaded_pdfs(uploaded_pdf_files, save_directory):
    """
    Process uploaded PDF files, save them locally, extract images based on their pages,
    and return a list of paths to the extracted images.

    Args:
        uploaded_pdf_files (list of UploadedFile): List of uploaded PDF file objects.
        save_directory (str): The directory where the PDF files and extracted images will be saved.

    Returns:
        list of str: List of paths to the extracted images.
    """
    uploaded_pdf_images = []

    if uploaded_pdf_files:
        for pdf_file in uploaded_pdf_files:
            # Save PDF files locally
            pdf_name = pdf_file.name[:-4]
            pdf_save_path = os.path.join(save_directory, f"{pdf_name}.pdf")
            with open(pdf_save_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            # Extract images from PDF based on their pages
            images = convert_from_bytes(pdf_file.read())

            for i, image in enumerate(images):
                image_name = f"{pdf_name}_page_{i+1}.png"
                image_save_path = os.path.join(save_directory, image_name)
                image.save(image_save_path, "PNG")
                uploaded_pdf_images.append(image_save_path)

    return uploaded_pdf_images


def process_youtube_analysis_and_summary(
        youtube_link, gemini_key, client_openai, entity_description=None, entity=None
) -> str:
    """
    Process analysis and summary of a YouTube video.

    Args:
        youtube_link (str): The URL of the YouTube video.
        gemini_key (str): The API key for Gemini.
        client_openai: The OpenAI client.
        entity_description (str, optional): Description of the entity being analyzed. Defaults to None.
        entity (str, optional): Name of the entity being analyzed. Defaults to None.

    Returns:
        str: The analysis and summary result for the YouTube video.
    """
    if youtube_link:
        audio_path, video_title = get_youtube(youtube_link)

        transcription = transcribe(
            audio_file_path=audio_path,
            client=client_openai,
            lang="fr")

        chunks = create_chunk_for_gemini(
            text=transcription.strip(),
            api_key=gemini_key,
            chunk_overlap=10,
            separator="\n\n")

        if len(chunks) == 1:
            youtube_final_prompt = audio_prompt.format(
                entity=entity or config.PLACEHOLDER_ENTITY,
                entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                audio_transcription=chunks[0])
            result_youtube_analysis_and_summary = generate_response_with_genai(
                youtube_final_prompt, gemini_key)
        else:
            youtube_relevant_text_temp = ""
            for chunk in chunks:
                prompt_for_extracting_relevant_text_for_youtube = audio_extract_text_prompt.format(
                    entity=entity or config.PLACEHOLDER_ENTITY,
                    entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                    audio_transcription=chunk)

                youtube_relevant_text_temp += generate_response_with_genai(
                    prompt_for_extracting_relevant_text_for_youtube, gemini_key)

            youtube_final_prompt = audio_prompt.format(
                entity=entity or config.PLACEHOLDER_ENTITY,
                entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                audio_transcription=youtube_relevant_text_temp)
            result_youtube_analysis_and_summary = generate_response_with_genai(
                youtube_final_prompt, gemini_key)

        return result_youtube_analysis_and_summary


def process_audio_analysis_and_summary(
        uploaded_audio_file, gemini_key, client_openai, entity_description, entity
) -> str:
    """
    Process analysis and summary of an uploaded audio file.

    Args:
        uploaded_audio_file: The uploaded audio file.
        gemini_key (str): The API key for Gemini.
        client_openai: The OpenAI client.
        entity_description (str): Description of the entity being analyzed.
        entity (str): Name of the entity being analyzed.

    Returns:
        str: The analysis and summary result for the uploaded audio file.
    """
    audio_path = save_uploaded_audio_file(
        uploaded_audio_file=uploaded_audio_file,
        save_directory=config.AUDIO_FILE)

    transcription = transcribe(
        audio_file_path=audio_path,
        client=client_openai,
        lang="fr")

    chunks = create_chunk_for_gemini(
        text=transcription.strip(),
        api_key=gemini_key,
        chunk_overlap=10,
        separator="\n\n")

    if len(chunks) == 1:
        audio_final_prompt = audio_prompt.format(
            entity=entity or config.PLACEHOLDER_ENTITY,
            entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
            audio_transcription=chunks[0])

        result_audio_analysis_and_summary = generate_response_with_genai(
            audio_final_prompt, gemini_key)

    else:
        audio_relevant_text_temp = ""
        for chunk in chunks:
            prompt_for_extracting_relevant_text_for_audio = audio_extract_text_prompt.format(
                entity=entity or config.PLACEHOLDER_ENTITY,
                entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                audio_transcription=chunk)

            audio_relevant_text_temp += generate_response_with_genai(
                prompt_for_extracting_relevant_text_for_audio, gemini_key)

        audio_final_prompt = audio_prompt.format(
            entity=entity or config.PLACEHOLDER_ENTITY,
            entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
            audio_transcription=audio_relevant_text_temp)

        result_audio_analysis_and_summary = generate_response_with_genai(
            audio_final_prompt, gemini_key)

    return result_audio_analysis_and_summary


def process_images_analysis_and_summary(
    uploaded_images_files, gemini_key, entity_description, entity
) -> str:
    """
    Process analysis and summary of uploaded images.

    Args:
        uploaded_images_files (List[UploadedFile]): List of uploaded image files.
        gemini_key (str): The API key for Gemini.
        entity_description (str): Description of the entity being analyzed.
        entity (str): Name of the entity being analyzed.

    Returns:
        str: The analysis and summary result for the uploaded images.
    """
    transcription = ""
    for image_file in uploaded_images_files:
        filename = image_file.name
        extension = filename.split(".")[-1].lower()
        if extension in ["jpg", "jpeg", "png"]:
            transcription += detect_text_blocks(image_file.getvalue())

    chunks = create_chunk_for_gemini(
        text=transcription.strip(),
        api_key=gemini_key,
        separator="\n\n")

    if len(chunks) == 1:
        images_final_prompt = images_prompt.format(
            entity=entity or config.PLACEHOLDER_ENTITY,
            entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
            extracted_text_images=chunks[0])

        result_images_analysis_and_summary = generate_response_with_genai(
            images_final_prompt, gemini_key)
    else:
        images_relevant_text_temp = ""
        for chunk in chunks:
            prompt_for_extracting_relevant_text_for_image = images_extract_text_prompt.format(
                entity=entity or config.PLACEHOLDER_ENTITY,
                entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                images_transcription=chunk)
            images_relevant_text_temp += generate_response_with_genai(
                prompt_for_extracting_relevant_text_for_image, gemini_key)

        images_final_prompt = images_prompt.format(
            entity=entity or config.PLACEHOLDER_ENTITY,
            entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
            extracted_text_images=images_relevant_text_temp)

        result_images_analysis_and_summary = generate_response_with_genai(
            images_final_prompt, gemini_key)

    return result_images_analysis_and_summary


def process_pdf_analysis_and_summary(
    uploaded_pdf_files, gemini_key, entity_description, entity
) -> str:
    """
    Process analysis and summary of uploaded PDF files.

    Args:
        uploaded_pdf_files (List[UploadedFile]): List of uploaded PDF files.
        gemini_key (str): The API key for Gemini.
        entity_description (str): Description of the entity being analyzed.
        entity (str): Name of the entity being analyzed.

    Returns:
        str: The analysis and summary result for the uploaded PDF files.
    """
    list_pdf_images_path = process_uploaded_pdfs(
        uploaded_pdf_files, config.IMAGE_FROM_PDF_FILE)
    transcription = ""
    for pdf_image_path in list_pdf_images_path:
        transcription += detect_text_blocks_pdf(pdf_image_path)

    chunks = create_chunk_for_gemini(
        text=transcription.strip(),
        api_key=gemini_key,
        chunk_overlap=10,
        separator="\n\n")

    if len(chunks) == 1:
        pdf_final_prompt = images_prompt.format(
            entity=entity or config.PLACEHOLDER_ENTITY,
            entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
            extracted_text_images=chunks[0])
        result_pdf_analysis_and_summary = generate_response_with_genai(
            pdf_final_prompt, gemini_key)
    else:
        pdf_images_relevant_text_temp = ""
        for chunk in chunks:
            prompt_for_extracting_relevant_text_for_pdf_image = images_extract_text_prompt.format(
                entity=entity or config.PLACEHOLDER_ENTITY,
                entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
                images_transcription=chunk)
            pdf_images_relevant_text_temp += generate_response_with_genai(
                prompt_for_extracting_relevant_text_for_pdf_image, gemini_key)

        pdf_final_prompt = images_prompt.format(
            entity=entity or config.PLACEHOLDER_ENTITY,
            entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
            extracted_text_images=pdf_images_relevant_text_temp)

        result_pdf_analysis_and_summary = generate_response_with_genai(
            pdf_final_prompt, gemini_key)

    return result_pdf_analysis_and_summary


def process_uploaded_video_analysis_and_summary(
    uploaded_video_file, client_openai, gemini_key, entity_description, entity
) -> str:
    """
    Process analysis and summary of uploaded video file.

    Args:
        uploaded_video_file (UploadedFile): Uploaded video file.
        client_openai (OpenAI): OpenAI client for API requests.
        gemini_key (str): The API key for Gemini.
        entity_description (str): Description of the entity being analyzed.
        entity (str): Name of the entity being analyzed.

    Returns:
        str: The analysis and summary result for the uploaded video file.
    """
    file_extension = uploaded_video_file.name.split(".")[-1].lower()
    if file_extension in ["mp4", "avi"]:
        video_bytes = uploaded_video_file.getvalue()
        with open(config.VIDEO_FILE, "wb") as f:
            f.write(video_bytes)
        st.session_state["sidebar_local_video_file"] = config.VIDEO_FILE

    audio_path = video_to_audio(
        st.session_state.sidebar_local_video_file,
        config.AUDIO_FILE_FROM_VIDEO)

    transcription = transcribe(
        audio_file_path=audio_path,
        client=client_openai,
        lang="fr")

    prompt_for_extracting_relevant_text_for_video_uploaded = audio_extract_text_prompt.format(
        entity=entity or config.PLACEHOLDER_ENTITY,
        entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
        audio_transcription=transcription)

    video_uploaded_relevant_text_temp = generate_response_with_genai(
        prompt_for_extracting_relevant_text_for_video_uploaded, gemini_key)

    video_final_prompt = audio_prompt.format(
        entity=entity or config.PLACEHOLDER_ENTITY,
        entity_description=entity_description or config.PLACEHOLDER_ENTITY_DESCRIPTION,
        audio_transcription=video_uploaded_relevant_text_temp)

    result_video_analysis_and_summary = generate_response_with_genai(
        video_final_prompt, gemini_key)

    return result_video_analysis_and_summary
