"""
utils_similarity.py

This module contains utility functions to clean and prepare embeddings for similarity calculations.

Modules:
- numpy: For handling numerical arrays.
- ast: For safely parsing string representations of lists.
- re: For regular expression operations.

Functions:
- clean_embeddings: Cleans and converts embedding strings into numpy arrays.
"""

import numpy as np
import ast
import re


def clean_embeddings(embedding_str: str) -> np.ndarray:
    """
    Clean and convert embedding strings into numpy arrays.

    Args:
        embedding_str (str): String representation of the embedding to clean.

    Returns:
        np.ndarray: Cleaned numpy array of the embedding.
    """
    
    # Ensure embedding strings are converted properly
    if isinstance(embedding_str, str):
        # Use regex to ensure proper formatting with commas
        cleaned_str = re.sub(r'(?<=\d)\s+(?=[\d-])', ', ', embedding_str.replace("\n", " "))
        return np.array(ast.literal_eval(cleaned_str))
    
    return embedding_str
