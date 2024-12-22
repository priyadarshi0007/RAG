import webvtt

def parse_vtt(file_path):
    """
    Parse a .vtt file and extract text content.
    """
    captions = []  # List to store the text from each caption
    for caption in webvtt.read(file_path):  # Iterates over each caption in the file
        captions.append(caption.text.strip())  # Add cleaned text from each caption to the list
    return " ".join(captions)  # Combine all captions into a single string