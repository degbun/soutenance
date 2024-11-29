
from typing import List

import google.generativeai as genai
from google.api_core import retry

# create chunck for gemini using  model.count_tokens, noticed that
# gemini-1.5-pro-latest use 1000000 tokens def create_chunk_for_gemini(


@retry.Retry(timeout=300.0)
def create_chunk_for_gemini(
    text: str,
    api_key: str,
    chunk_size: int = 50000,
    chunk_overlap: int = 10,
    separator: str = "\n\n'"
) -> List[str]:
    """
    Splits a large text into smaller chunks suitable for processing with the Generative AI (Gemini) model.

    Args:
        text (str): The input text to be segmented into chunks.
        api_key (str): The API key for accessing the Generative AI service.
        chunk_size (int, optional): The maximum size limit (in tokens) for each chunk. Defaults to 2200.
        chunk_overlap (int, optional): The number of overlapping tokens between adjacent chunks to maintain context continuity. Defaults to 10.
        separator (str, optional): The separator used to split the text into sequences. Defaults to "\n\n'".

    Returns:
        List[str]: A list of segmented chunks of text.
    """

    # Check if the input text is empty
    if not text.strip():
        return []  # Return empty list for empty input

    # Configure the Generative AI service (uncomment if using genai library)
    genai.configure(api_key=api_key)

    # Create a single GenerativeModel instance for the Gemini model (uncomment
    # if using genai library)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    # Count tokens in the entire text
    total_tokens = model.count_tokens(text.strip()).total_tokens

    if total_tokens < chunk_size:
        return [text.strip()]

    else:

        chunks = []

        # Split the input text into sequences using the specified separator
        sequences = text.split(separator)

        chunk = " "  # Initialize an empty string to build each chunk

        for seq in sequences:
            # Check if the sequence is empty or contains only whitespace

            if not seq.strip():
                continue
            print(seq)
            print(sequences.index(seq))
            # Check if adding the current sequence to the chunk would exceed the
            # maximum chunk size
            if (model.count_tokens(chunk).total_tokens) + \
                    (model.count_tokens(seq.strip()).total_tokens) < chunk_size:
                # If not, add the sequence to the current chunk along with the
                # separator
                chunk += seq + separator
            else:
                # If adding the sequence would exceed the maximum chunk size,
                # add the current chunk to the list of chunks after stripping any
                # trailing whitespace
                chunks.append(chunk.strip())
                # Update the current chunk by keeping only the last few sequences
                # to maintain overlap
                chunk = separator.join(chunk.split(separator)[-chunk_overlap:])

        # Check if there are any remaining tokens in the last chunk
        if int(model.count_tokens(chunk.strip()).total_tokens) < chunk_size:
            # If the remaining tokens are within the maximum chunk size, add the
            # chunk to the list of chunks
            chunks.append(chunk.strip())

        return chunks  # Return the list of segmented chunks


def get_relevant_content(
        text_extract: str,
        relevant_content: str,
        window_size: int,
        separator: str = "\n\n") -> str:
    """
    Extracts relevant content from a larger text based on extracted text.

    Args:
        text_extract (str): The extracted text.
        relevant_content (str): The larger text containing the relevant content.
        window_size (int): The window size for extracting surrounding content.
        separator (str, optional): The separator used to split the text. Defaults to "\n\n".

    Returns:
        str: The relevant content along with surrounding context.

    Example:
        >>> extract = "relevant text"
        >>> content = "This is some relevant text.\n\nThis is some other text."
        >>> get_relevant_content(extract, content, 1)
        'This is some relevant text.\n\nThis is some other text.'
    """
    # Split the extracted text and relevant content into segments
    extract = text_extract.split(separator)
    content = relevant_content.split(separator)

    # Find indices of relevant content in the larger text
    index = []
    for text in extract:
        if text not in content:
            continue
        idx = content.index(text)
        index += [max(0, idx - i) for i in range(1, window_size + 1)] + [
            min(len(content) - 1, idx + i) for i in range(1, window_size + 1)
        ]

    # Sort and remove duplicates from the index list
    index = sorted(set(index))

    # Join relevant content along with surrounding context
    return separator.join([content[i] for i in index])


@retry.Retry(timeout=7200000)
def generate_response_with_genai(prompt: str, api_key: str) -> str:
    """
    Generate a response using the Generative AI model.

    Args:
        prompt (str): The input text for the model.
        api_key (str): The API key for the Generative AI service.

    Returns:
        str: The generated response.
    """
    # Configure the Generative AI service with the provided API key
    genai.configure(api_key=api_key)

    # Create a single GenerativeModel instance for the Gemini model
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    # Generate the response using the prompt
    response = model.generate_content(prompt)

    return response.text
