import webvtt

def parse_vtt(file_path):
    """
    Parse a .vtt file and extract text content.
    """
    captions = []
    for caption in webvtt.read(file_path):
        captions.append(caption.text.strip())
    return " ".join(captions)

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks for vectorization.
    """
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks