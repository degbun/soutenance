

import cv2
import layoutparser as lp
import numpy as np


# extract text from image with layoutparser model prima
def detect_text_blocks(image_path) -> str:
    """
    Extracts "tout et tout" (everything and everything) of the textual content from an image.

    This function leverages Layout Parser (LP) for layout detection and Tesseract for Optical Character Recognition (OCR) to locate and extract all text blocks within an image. It then returns the combined extracted text, ensuring it captures "tout et tout" of the textual information present.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A single string containing "tout et tout" of the extracted text from the image. This text is separated by newlines and has leading/trailing whitespace and any extraneous control characters removed.

    Raises:
        FileNotFoundError: If the image file cannot be loaded.
    """
    # Load the image

    nparr = np.frombuffer(image_path, np.uint8)
    # Decode numpy array to image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the layout detection model
    model = lp.models.Detectron2LayoutModel(
        'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.8],
        label_map={
            1: "TextRegion",
            2: "ImageRegion",
            3: "TableRegion",
            4: "MathsRegion",
            5: "SeparatorRegion",
            6: "OtherRegion"})

    # Detect layout
    layout = model.detect(image_rgb)

    # Filter text blocks
    text_blocks = lp.Layout([b for b in layout if b.type == 'TextRegion'])

    # Get image dimensions
    h, w = image.shape[:2]

    # Define interval for left half of the image
    left_interval = lp.Interval(0, w / 2 * 1.05, axis='x').put_on_canvas(image)

    # Filter text blocks for left and right halves
    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
    right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
    right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    # Combine left and right blocks and add index
    text_blocks = lp.Layout([b.set(id=idx)
                            for idx, b in enumerate(left_blocks + right_blocks)])

    # Initialize Tesseract OCR agent
    ocr_agent = lp.TesseractAgent(languages='eng')

    # Initialize variable to store all extracted text
    all_text = ""

    # Extract text from text blocks
    for block in text_blocks:
        # Crop and preprocess image segment
        segment_image = (block.pad(left=5, right=5, top=5, bottom=5)
                         .crop_image(image_rgb))

        # Perform OCR
        text = ocr_agent.detect(segment_image).strip()

        # Append extracted text to the result string with a newline separator
        all_text += text + "\n\n"

    # Return all extracted text
    return all_text.strip()


def detect_text_blocks_pdf(image_path) -> str:
    """
    Extracts "tout et tout" (everything and everything) of the textual content from an image.

    This function leverages Layout Parser (LP) for layout detection and Tesseract for Optical Character Recognition (OCR) to locate and extract all text blocks within an image. It then returns the combined extracted text, ensuring it captures "tout et tout" of the textual information present.
    This function is to process the image extracted from our uploaded pdf

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A single string containing "tout et tout" of the extracted text from the image. This text is separated by newlines and has leading/trailing whitespace and any extraneous control characters removed.

    Raises:
        FileNotFoundError: If the image file cannot be loaded.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # Convert BGR to RGB
    image_rgb = image[..., ::-1]

    # Load the layout detection model
    model = lp.models.Detectron2LayoutModel(
        'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.8],
        label_map={
            1: "TextRegion",
            2: "ImageRegion",
            3: "TableRegion",
            4: "MathsRegion",
            5: "SeparatorRegion",
            6: "OtherRegion"})

    # Detect layout
    layout = model.detect(image_rgb)

    # Filter text blocks
    text_blocks = lp.Layout([b for b in layout if b.type == 'TextRegion'])

    # Get image dimensions
    h, w = image.shape[:2]

    # Define interval for left half of the image
    left_interval = lp.Interval(0, w / 2 * 1.05, axis='x').put_on_canvas(image)

    # Filter text blocks for left and right halves
    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
    right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
    right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    # Combine left and right blocks and add index
    text_blocks = lp.Layout([b.set(id=idx)
                            for idx, b in enumerate(left_blocks + right_blocks)])

    # Initialize Tesseract OCR agent
    ocr_agent = lp.TesseractAgent(languages='eng')

    # Initialize variable to store all extracted text
    all_text = ""

    # Extract text from text blocks
    for block in text_blocks:
        # Crop and preprocess image segment
        segment_image = (block.pad(left=5, right=5, top=5, bottom=5)
                         .crop_image(image_rgb))

        # Perform OCR
        text = ocr_agent.detect(segment_image).strip()

        # Append extracted text to the result string with a newline separator
        all_text += text + "\n\n"

    # Return all extracted text
    return all_text.strip()
