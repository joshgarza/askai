from flask import Flask, request, jsonify
from flask_cors import CORS
from ask_ai import ask_ai, construct_index
import time

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
    query = data.get("query")
    response = ask_ai(query)

    # seconds passed since epoch
    seconds = time.time()

    # convert the time in seconds since the epoch to a readable format
    local_time = time.ctime(seconds)

    print("Local time:", local_time)

    # log request and response to logger.txt
    f = open("logger.txt", "a")
    f.write(f"Query: {query}\nResponse: {response}\nDate: {local_time}\n\n")
    f.close()

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run()
