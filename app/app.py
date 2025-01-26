"""
app.py

This script creates a Streamlit web application for the book recommendation system.
It dynamically downloads the preprocessed `embeddings.csv` file from Google Drive,
loads the data into a Pandas DataFrame, and enables users to find similar books
based on cosine similarity of embeddings.

Modules:
- streamlit: For building the web application interface.
- pandas: For handling and manipulating book data.
- numpy: For numerical operations, specifically handling embeddings.
- gdown: For downloading large files from Google Drive, handling confirmation warnings.
- src.generate_similarity: Contains the `find_similar_books` function to calculate book similarities.

Usage:
Run the app locally using:
$ streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import gdown
import sys
import os
import numpy as np
import ast
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.generate_similarity import find_similar_books



@st.cache_data

def download_file_from_google_drive(url, output_path):
    """
    Downloads a large file from Google Drive using `gdown`.

    Args:
        url (str): Shared link from Google Drive.
        output_path (str): Path where the downloaded file will be saved.
    """
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"File successfully downloaded to: {output_path}")
    except Exception as e:
        print(f"Error downloading the file: {e}")


def load_data() -> pd.DataFrame:
    """
    Load the preprocessed book data from `embeddings_app.csv`.

    Returns:
        pd.DataFrame: DataFrame with book data and embeddings.
    """

    # Load preprocessed data
    df = pd.read_csv('data/processed/embeddings_app.csv')

    # Clean and convert embedding strings into numpy arrays directly
    def parse_embedding(embedding_str):
        if isinstance(embedding_str, str):
            # Ensure proper formatting with commas
            cleaned_str = re.sub(r'(?<=\d)\s+(?=[\d-])', ', ', embedding_str.replace("\n", " "))
            return np.array(ast.literal_eval(cleaned_str))
        return embedding_str

    # Apply cleaning inline
    df['embedding'] = df['embedding'].apply(parse_embedding)

    # Drop rows with invalid embeddings
    df = df.dropna(subset=['embedding'])

    return df
    

# Streamlit app layout
def streamlit_app():
    """
    Main function to run the Streamlit app for the book recommendation system.
    """

    # Page configuration
    st.set_page_config(
        page_title="Book Recommendation System",
        page_icon="📚",
        layout="wide",
    )

    # App title and description
    st.title("📚 Book Recommendation System")
    st.markdown(
        """
        **Discover books similar to your favorites**  
        Explore personalized recommendations based on your preferences.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Options")
    st.sidebar.markdown("Configure your search:")
    
    # Download data
    google_drive_url = "https://drive.google.com/uc?id=1_CAPfen20aaJ0MjqbWRI4twzLt2F7XU6&export=download"
    output_file = os.path.join("data", "processed", "embeddings_app.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    download_file_from_google_drive(google_drive_url, output_file)
    if not os.path.exists(output_file):
        st.error(f"The file {output_file} was not downloaded. Check the URL or permissions.")
        return

    # Load data
    df = load_data()
    if df.empty:
        st.error("Data could not be loaded. Please check the logs for more details.")
        return
    
    # Sidebar options
    st.sidebar.header("Options")
    st.sidebar.markdown("Configure your search:")
    book_title = st.sidebar.text_input("Enter a book title:", value="The Hunger Games")
    top_n = st.sidebar.slider("Number of recommendations:", 1, 5, 3)
    
    # Main content
    if st.sidebar.button("🔍 Find Similar Books"):
        similar_books = find_similar_books(book_title, df, top_n=top_n)
        
        st.subheader(f"Books similar to **{book_title}**")
    
        # Display recommendations as plain text lines
        for _, row in similar_books.iterrows():
            st.markdown(f"- **{row['book_title']}** by {row['book_authors']}")
            st.markdown(f"*{row['book_desc']}*")  # Display the book description
            st.markdown("---")  # Add a separator between books
    else:
        st.markdown("Select a book and click 'Find Similar Books' to get recommendations.")
            

if __name__ == "__main__":
    streamlit_app()


