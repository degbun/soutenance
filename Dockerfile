FROM python:3.10
# Expose the port for Streamlit app
EXPOSE 8501

# Update package lists and install dependencies efficiently
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create data folder structure with subdirectories
WORKDIR /app
RUN mkdir -p data/youtube_data data/video_data data/audio_data data/pdf_to_images_data data/images_data

# Copy requirements.txt securely
COPY requirements.txt /tmp/requirements.txt

# Upgrade pip and install dependencies without cache
RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt && rm -r /tmp/requirements.txt

# Install detectron2 from GitHub
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"

# Copy application code and configuration
COPY ./src src
COPY ./config.py config.py
COPY ./main.py main.py



# Entrypoint command to run the Streamlit app
ENTRYPOINT [ "streamlit", "run", "main.py" ]
