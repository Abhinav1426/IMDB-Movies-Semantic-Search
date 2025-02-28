import itertools
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorDB:
    def __init__(self, api_key, index_name):
        self.api_key = api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.api_key)
        self.index_model = None

    def create_new_index(self, dimension, metric, cloud, region):
        if not self.pc.has_index(self.index_name):
            model = self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            self.get_index_model()
            print("Index created successfully!")
            print(model)
        else:
            print("Index already exists!")

    def get_index_model(self):
        self.index_model = self.pc.Index(self.index_name)
        return self.index_model
    def add_records(self, df):
        records = []
        for i, row in df.iterrows():
            records.append({
                "id": str(i),
                "values": row['embedding'],
                "metadata": {
                    "Series_Title": row['Series_Title'],
                    "Genre": row['Genre'],
                    "Director": row['Director'],
                    "Year": row['Released_Year'],
                    "Rating": row['IMDB_Rating'],
                    "Overview": row['Overview'],
                    "Star1": row['Star1'],
                    "Star2": row['Star2'],
                    "Star3": row['Star3'],
                    "Star4": row['Star4'],
                    "PosterLink": row['Poster_Link']
                }
            })
        self.insert_records_batch(records)
    @staticmethod
    def chunks(iterable, batch_size=200):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def insert_records_batch(self, records):
        for ids_vectors_chunk in self.chunks(records, batch_size=200):
            self.index_model.upsert(vectors=ids_vectors_chunk)