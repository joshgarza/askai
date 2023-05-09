from flask import Flask, request, jsonify
from flask_cors import CORS
from ask_ai import ask_ai, construct_index

app = Flask(__name__)
CORS(app)

construct_index("context_data/data")


@app.route("/", methods=["GET"])
def home():
    return "Welcome to the AI chatbot API!"


@app.route("/test", methods=["GET"])
def test():
    return "This is a test..."


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    print(data)
    query = data.get("query")
    response = ask_ai(query)
    print(response)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run()
