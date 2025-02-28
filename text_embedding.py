import pandas as pd
from sentence_transformers import SentenceTransformer

class Text_Processer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = None

    def load_data(self, path):
        self.df = pd.read_csv(path)
        print("Data loaded successfully!")

    def filter_and_combine_text(self, cols):
        self.df = self.df[cols]
        self.df['combined_text'] = self.df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("Text combined successfully!")

    def get_embeddings(self):
        self.df['embedding'] = self.df['combined_text'].apply(lambda text: self.model.encode(text))
        return self.df
