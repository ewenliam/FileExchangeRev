from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/file-upload', methods=['POST'])
def file_upload():
    data = request.json
    user_id = data.get('userId')
    file_path = data.get('filePath')
    
    # Handle the file upload notification
    # This can include saving details to a database or notifying users
    
    return jsonify({'message': 'File upload received'}), 200

if __name__ == '__main__':
    app.run(port=5001)
