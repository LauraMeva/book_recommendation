"""
generate_embeddings.py

This script loads preprocessed book data, generates embeddings for the book descriptions,
and saves the DataFrame with embeddings for future use.

Modules:
- pandas: For data manipulation.
- libs.utils_embeddings: Contains the embedding generation utilities.

Usage:
Run this script directly to generate and save embeddings:
$ python generate_embeddings.py
"""

import pandas as pd
from loguru import logger
from libs.utils_embeddings import add_embeddings


def main():
    """
    Main function to load preprocessed data, generate embeddings,
    and save the updated DataFrame.
    """
    logger.info("Starting the embedding generation process.")

    # Load preprocessed data
    try:
        logger.info("Loading preprocessed data from '../data/processed/cleaned_books.csv'.")
        df = pd.read_csv('../data/processed/cleaned_books.csv')
    except FileNotFoundError:
        logger.error("File '../data/processed/cleaned_books.csv' not found. Please check the path.")
        return
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading data: {e}")
        return

    # Generate embeddings
    try:
        logger.info("Generating embeddings for book descriptions.")
        df = add_embeddings(df)
    except Exception as e:
        logger.exception(f"An error occurred while generating embeddings: {e}")
        return

    # Save the DataFrame with embeddings
    try:
        logger.info("Saving the DataFrame with embeddings to '../data/processed/books_with_embeddings.csv'.")
        df.to_csv('../data/processed/embeddings.csv', index=False)
        logger.success("Embeddings saved successfully!")
    except Exception as e:
        logger.exception(f"An error occurred while saving the embeddings: {e}")
        return


# Entry point
if __name__ == "__main__":
    main()