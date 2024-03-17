import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from process_video import process_video
from translate_lsp import evaluate_model

app = Flask(__name__)

@app.route('/')
def hello():
    return '¡Hola, mundo!'

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    file_name = secure_filename(video_file.filename)
    tmp_file = os.path.join('tmp', file_name)
    video_file.save(tmp_file)
     
    resp = evaluate_model(process_video(tmp_file))
    
    return " ".join(resp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

