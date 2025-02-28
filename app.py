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
    div[data-testid="InputInstructions"] > span:nth-child(1) {
    visibility: hidden;
    }
    div[data-testid="stButton"] {
        margin-top: 27px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    '<div class="header"><h1>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Semantic Search &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp IMDB Movies Dataset🎬</h1></div>',
    unsafe_allow_html=True)


c1,c2 = st.columns((4,1))
with c1:
    query = st.text_input("Enter your Search Query 🔍", placeholder="Describe the type of movie you're looking for...", key="Type Here ...")
with c2:
    click = st.button("Search 🚀")
add_slider = st.slider("Select the number of recommendations", 1, 25, 5)

if click:
    if query:
        # Generate embedding for the query
        query_embedding = model.encode(query)
        results = index.query(vector=query_embedding.tolist(), top_k=add_slider, include_metadata=True)

        st.subheader("Recommended Movies:")
        for match in results['matches']:
            metadata = match['metadata']
            poster_url = metadata.get('PosterLink', "https://static.vecteezy.com/system/resources/previews/007/459/267/non_2x/set-of-cinema-icons-movie-design-elements-with-a-cartoon-concept-illustration-free-vector.jpg")
            movie_card = f"""
            <div class="movie-card">
                <img src="{poster_url}" alt="Movie Poster" class="movie-poster">
                <div class="movie-details">
                    <div class="movie-title">{metadata.get('Series_Title', 'Unknown Title')}</div>
                    <div class="movie-detail"><strong>Year:</strong> {metadata.get('Year', 'N/A')}</div>
                    <div class="movie-detail"><strong>Genre:</strong> {metadata.get('Genre', 'N/A')}</div>
                    <div class="movie-detail"><strong>Rating:</strong> {metadata.get('Rating', 'N/A')} ⭐️</div>
                    <div class="movie-detail"><strong>Director:</strong> {metadata.get('Director', 'N/A')}</div>
                    <div class="movie-detail"><strong>Starring:</strong> {metadata.get('Star1', '')}, {metadata.get('Star2', '')}, {metadata.get('Star3', '')}, {metadata.get('Star4', '')}</div>
                    <div class="movie-detail"><strong>Description:</strong> {metadata.get('Overview', 'No description available')}</div>
                </div>
            </div>
            """
            st.markdown(movie_card, unsafe_allow_html=True)
    else:
        st.write("⚠️ Please enter a search query")