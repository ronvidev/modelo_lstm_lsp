import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from process_video import process_video
from evaluate_model import evaluate_model

app = Flask(__name__)

@app.route('/')
def hello():
    return 'LSP Translate'

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    file_name = secure_filename(video_file.filename)
    root_path = os.path.dirname(os.path.abspath(__file__))
    tmp_file = os.path.join(root_path, 'tmp', file_name)
    video_file.save(tmp_file)
    
    video_processed = process_video(tmp_file)
    resp = evaluate_model(video_processed)
    resp = [r.upper() for r in resp][::-1]
    
    return " - ".join(resp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

