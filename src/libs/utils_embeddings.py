"""
utils_embeddings.py

This module contains utility functions to generate embeddings for book descriptions
using a pre-trained language model from the Hugging Face Transformers library.

Modules:
- pandas: For handling the DataFrame.
- torch: For PyTorch tensor manipulation and model inference.
- transformers: For loading the tokenizer and model.

Functions:
- embed_text: Encodes a single text into a numerical embedding.
- add_embeddings: Applies the embedding function to all descriptions in a DataFrame.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# Load the model and tokenizer
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_text(text: str) -> np.ndarray:
    """
    Generate a numerical embedding for a given text using a pre-trained model.

    Args:
        text (str): The input text to be embedded.

    Returns:
        numpy.ndarray: A numerical vector representing the text embedding.
    """
     
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model output (we'll use the embeddings of the [CLS] token for simplicity)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
    
    return embeddings.squeeze().numpy()  # Convert to numpy array for easy handling


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate embeddings for the 'book_desc' column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'book_desc' column with text descriptions.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'embedding' column.
    """

    # Apply the embedding function to the 'book_desc' column
    df['embedding'] = df['book_desc'].apply(embed_text)
    
    return df