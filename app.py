import streamlit as st
from text_embedding import Text_Processer
from pinecone_vector_db import PineconeVectorDB

model = Text_Processer().model
apiKey, index_name = "pcsk_7UwC1c_G2ePdtPHYRBGDn3EZwhZjowdtbdKyxQ1TYTbHzZmcUS1LGkmR8HcBbXFqeU8VTd", "imdb-movie-search"
index = PineconeVectorDB(apiKey, index_name).get_index_model()

st.markdown(
    """
    <style>

    /* Header styling */
    .header {
        text-align: center;
        padding: 25px;
    }
    /* Movie card styling */
    .movie-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: flex-start;
    }
    .movie-poster {
        width: 170px;
        height: auto;
        border-radius: 4px;
        margin-right: 16px;
    }
    .movie-details {
        flex: 1;
    }
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 8px;
        color: #333333;
    }
    .movie-detail {
        margin-bottom: 4px;
        color: #555555;
    }
    div[data-baseweb="base-input"]{ 
      width: 100%;
      border: 3px solid #ccc;
      border-radius: 10px;
      font-size: 20px;
      outline: none;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    '<div class="header"><h1>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Semantic Search &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp IMDB Movies Dataset</h1></div>',
    unsafe_allow_html=True)

query = st.text_input("Search Movies 🔎",placeholder="Describe the type of movie you're looking for...")

if query:
    # Generate embedding for the query
    query_embedding = model.encode(query)
    results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)

    st.subheader("Recommended Movies:")
    for match in results['matches']:
        metadata = match['metadata']
        poster_url = metadata.get('PosterLink', "https://via.placeholder.com/150x220.png?text=No+Image")
        movie_card = f"""
        <div class="movie-card">
            <img src="{poster_url}" alt="Movie Poster" class="movie-poster">
            <div class="movie-details">
                <div class="movie-title">{metadata.get('Series_Title', 'Unknown Title')}</div>
                <div class="movie-detail"><strong>Year:</strong> {metadata.get('Year', 'N/A')}</div>
                <div class="movie-detail"><strong>Genre:</strong> {metadata.get('Genre', 'N/A')}</div>
                <div class="movie-detail"><strong>Director:</strong> {metadata.get('Director', 'N/A')}</div>
                <div class="movie-detail"><strong>Starring:</strong> {metadata.get('Star1', '')}, {metadata.get('Star2', '')}, {metadata.get('Star3', '')}, {metadata.get('Star4', '')}</div>
                <div class="movie-detail"><strong>Description:</strong> {metadata.get('Overview', 'No description available')}</div>
            </div>
        </div>
        """
        st.markdown(movie_card, unsafe_allow_html=True)
