import re
from transformers import AutoTokenizer

def preprocess_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    # Add special tokens for sentence segmentation
    text = text.replace(".", ".[SEP]").replace("!", "![SEP]").replace("?", "?[SEP]")

    # Tokenize text using a pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokens = tokenizer.tokenize(text)

    # Convert tokens back to text
    preprocessed_text = " ".join(tokens)

    return preprocessed_text
