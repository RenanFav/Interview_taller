import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def processing_text(text):
    text = text.lower()
    return text



df = pd.read_csv("customer_service_conversations.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')
df['agent process'] = df['Agent Response'].apply(processing_text)
embeddings = model.encode(df['agent process'])

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def top_3(query,n=3):
    query_model = model.encode(query)
    size,index = index.search(query,n)
    rp_1= df.iloc[index[0]['agent response']]
    rp_2 = size[0]
    return rp_1, rp_2
