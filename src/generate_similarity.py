"""
generate_similarity.py

This script computes book similarity based on embeddings. It allows you to find
books similar to a given book using cosine similarity.

Modules:
- numpy: For numerical operations.
- pandas: For data manipulation.
- sklearn.metrics.pairwise: For calculating cosine similarity.
- libs.utils_similarity: Contains utilities for cleaning and preparing embeddings.

Usage:
- Direct execution:
    $ python generate_similarity.py
"""

import numpy as np
import pandas as pd
from loguru import logger
from libs.utils_similarity import clean_embeddings
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_books(book_title: str, data: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Find books similar to a given book based on cosine similarity of embeddings.

    Args:
        book_title (str): Title of the book to find similarities for.
        data (pd.DataFrame): DataFrame containing book titles and embeddings.
        top_n (int): Number of similar books to return.

    Returns:
        pd.DataFrame: DataFrame containing similar book titles, authors, and their similarity scores.
    """
    if book_title not in data['book_title'].values:
        logger.error(f"Book title '{book_title}' not found in the dataset.")
        return pd.DataFrame(columns=['book_title', 'book_authors', 'similarity'])

    # Get the embedding of the target book
    target_embedding = data.loc[data['book_title'] == book_title, 'embedding'].values[0]
    embeddings = np.vstack(data['embedding'].values)
    
    # Compute cosine similarity
    similarities = cosine_similarity([target_embedding], embeddings).flatten()
    
    # Get the top_n most similar books (excluding the book itself)
    similar_indices = similarities.argsort()[::-1][1:top_n + 1]
    similar_books = data.iloc[similar_indices]
    
    # Add similarity scores to the DataFrame
    similar_books = similar_books[['book_title', 'book_authors', 'book_desc']].copy()
    similar_books['similarity'] = similarities[similar_indices]
    
    return similar_books


def main(book_title: str = "1984", n: int = 3):
    """
    Main function to load data, clean embeddings, and find similar books.

    Args:
        book_title (str): Title of the book to find similarities for (default: "1984").
        n (int): Number of similar books to return (default: 3).
    """
    logger.info("Starting the similarity computation process.")

    # Load data
    logger.info("Loading data from 'data/processed/embeddings.csv'.")
    try:
        df = pd.read_csv('data/processed/embeddings.csv')
    except FileNotFoundError:
        logger.error("File 'data/processed/embeddings.csv' not found. Please check the path.")
        return

    # Clean embeddings if necessary
    logger.info("Cleaning embedding strings.")
    df['embedding'] = df['embedding'].apply(clean_embeddings)

    # Similarity function
    logger.info(f"Finding books similar to '{book_title}'.")
    similar_books = find_similar_books(book_title, df, top_n=n)
    
    if similar_books.empty:
        logger.info(f"No similar books found for '{book_title}'.")
    else:
        print(f"Books similar to '{book_title}':")
        print(similar_books)


# Entry point
if __name__ == "__main__":
    main()
