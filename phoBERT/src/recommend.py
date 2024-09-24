import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")


def get_embedding(text):
    max_length = 512  # Set this based on your model's capabilities
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def create_embeddings(df):
    df['subject_embedding'] = df['subjectCode'].apply(lambda x: get_embedding(x['subjectName']))
    df['author_embedding'] = df['author'].apply(lambda x: get_embedding(x['name']))

    # Combine embeddings
    df['combined_embedding'] = df.apply(
        lambda row: np.mean([row['subject_embedding'], row['author_embedding']], axis=0),
        axis=1
    )
    return df


def recommend_books(query_subject, query_author, books_df, top_n=5):
    subject_embedding = get_embedding(query_subject)
    author_embedding = get_embedding(query_author)

    # Combine query embeddings
    query_combined = np.mean([subject_embedding, author_embedding], axis=0)

    # Calculate cosine similarities
    similarities = cosine_similarity([query_combined], np.array(books_df['combined_embedding'].tolist()))
    similar_indices = similarities[0].argsort()[-top_n - 1:-1][::-1]  # Get top N similar indices

    return books_df['title'].iloc[similar_indices].tolist()
