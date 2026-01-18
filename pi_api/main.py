import base64
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return jsonify({'endpoints': [
        "/capture"
    ]}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'biometric-capture-api',
        'version': '1.0.0'
    }), 200

@app.route('/capture', methods=['POST'])
def request_capture():
    data = request.get_json()
    user = data['user']

    #TODO: CAPTURE IMAGE --> SEND VECTOR
    vector = base64.b64encode('0'.encode())

    return jsonify({
        'user': user,
        'vector': vector,
        'protocol': 'palm-vein',
    })

if (__name__ == '__main__'):
    app.run('0.0.0.0', port=8443)