import warnings
warnings.filterwarnings("ignore")

from text_embedding import Text_Processer
from pinecone_vector_db import PineconeVectorDB



if __name__ == "__main__":
    print("Starting the process...")
    # Initialize text processor and data processor
    text_processor = Text_Processer()
    text_processor.load_data('data/imdb_top_1000.csv')
    print("Data loaded successfully!")
    text_processor.filter_and_combine_text(['Series_Title', 'Genre', 'Overview','Released_Year','IMDB_Rating', 'Director','Star1','Star2','Star3','Star4','Poster_Link'])
    df = text_processor.get_embeddings()
    print("Embeddings generated successfully!")
    # Initialize Pinecone Vector Database
    apiKey, index_name = "pcsk_7UwC1c_G2ePdtPHYRBGDn3EZwhZjowdtbdKyxQ1TYTbHzZmcUS1LGkmR8HcBbXFqeU8VTd", "imdb-movie-search"
    pinecone = PineconeVectorDB(apiKey, index_name)
    embedding_dimension = len(df['embedding'].iloc[0])
    print("Embedding dimension:", embedding_dimension)
    pinecone.create_new_index(dimension=embedding_dimension, metric='cosine', cloud='aws', region='us-east-1')
    pinecone.get_index_model()
    # pinecone.add_records(df)
    print("Records added successfully!")
    print("Process completed successfully!")

    query = "thriller Christopher Nolan movies"
    query_embedding = text_processor.model.encode(query)

    results = pinecone.index_model.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    print("Query results:")
    print(results)
