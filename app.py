from flask import Flask, request, jsonify
import preprocess as pre
import recommend as rec

# Initialize Flask app
app = Flask(__name__)

# Load and create embeddings
books_df = pre.load_books('/src/books.json')
books_df = rec.create_embeddings(books_df)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query_subject = data.get('subject', '')
    query_author = data.get('author', '')

    if not query_subject and not query_author:
        return jsonify({'error': 'Please provide a subject or author.'}), 400

    recommended_titles = rec.recommend_books(query_subject, query_author, books_df)
    return jsonify({'recommended_books': recommended_titles})


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
