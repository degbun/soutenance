# chat-with-image-video-pdf-projet ![Status](https://img.shields.io/badge/status-stable-brightgreen) ![Python Version](https://img.shields.io/badge/python-3.10.12-blue) ![Streamlit Version](https://img.shields.io/badge/Streamlit-1.28.2-brightgreen) ![Jupyter](https://img.shields.io/badge/Jupyter-yes-brightgreen)
This projet leverages Streamlit to analyze various types of media content, such as audio, images, PDFs, videos,newspaper, and YouTube links, utilizing OpenAI, Google AI Studio APIs, layoutParser library for transcription and content extraction


## Project Structure
```bash
streamlit/
├── config.toml/              # This config file sets server and runner settings for deployment.

data/
├── video_data/               # This folder is used to store the data(audio from the video and the video itself) of the uploaded local video
├── youtube_data/             # This folder is used to store the data(audio from the Youtube video and  the Youtube video itself) of the uploaded Youtube video
├── images_data/              # This folder is used to store the data(images) of the uploaded local images
├── pdf_to_images_data/       # This folder is used to store the data(images extracted fom the pdf pages and the pdfs  themself) of the uploaded pdfs
└── audio_data/               # This folder is used to store the audio data of the uploaded audio

src/
├── asr.py                    # This file contains the function that leverages Automatic speech recognition to extract text from audio
├── llm.py                    # This file contains functions for text chunking, content extraction, and response generation using a Generative AI model.
├── ocr_with_pytesseract.py   # This file contains function that  extracts all textual content from images by detecting text blocks and performing Optical Character Recognition (OCR),it utilize layoutPaser library  .
├── prompts.py                # This file  contains prompts for AI
└── utils.py                  # This file contains functions for various tasks such as converting video to audio, downloading YouTube videos, saving images, displaying results, clearing folder contents, and generating unique name

config.py                     # This configuration file contains placeholders for various API keys, links, and descriptions, along with paths for different types of files such as audio, video, images, and PDF conversions.
Dockerfile                    # Dockerfile for deployment
main.py                       # Main program file
requirements.txt              # File specifying project dependencies
```
## Installation
To install and run this project, you need to follow these steps :

#### Step 0: Clone the project
- On Lunix/unbuntu
```bash
git clone https://github.com/degbun/chat-with-image-video-projet-data.git
# enter in the cloned folder
cd  chat-with-image-video-projet-data
```


#### Step 1: Create a virtual environnement
- On Lunix/unbuntu
```bash
python3 -m venv venv
```
- On Windows
```bash
python -m venv venv
```

#### Step 2: Activate the a virtual environnement
- On Lunix/unbuntu
```bash
source venv/bin/activate
```
- On Windows
```bash
venv\Scripts\activate
```

#### Step 3: install the requirements

```bash
pip install -r requirements.txt
```

#### Step 4: run the app

```bash
streamlit run main.py
```






