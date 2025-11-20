from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
from chatbot_core import UniversityChatbot

app = Flask(__name__)
CORS(app)

print("Initializing Chatbot...")
bot = UniversityChatbot()
print("BU Buddy Ready!")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    if not message: return jsonify({'error': 'Empty message'}), 400

    response_data = bot.get_response(message)
    return jsonify({
        'response': response_data['response'],
        'data': response_data['data'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(port=8000, debug=True)