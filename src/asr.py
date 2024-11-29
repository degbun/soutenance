def transcribe(
    audio_file_path: str,
    client,
    lang: str,
) -> str:
    """
    Transcribe the audio content of a video file.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.
        client: The client object for accessing the transcription service.
        lang (str): The language of the audio content in the video.

    Returns:
        str: The transcription result.

    Example:
        >>> client = some_transcription_client_object
        >>> transcribe("audio.mp3", client, "en-US")
    """
    # Open the video file in binary read mode
    audio_file = open(audio_file_path, "rb")

    # Create the transcription using the client object
    return client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language=lang,
        response_format="vtt",
        temperature=0,
    )
