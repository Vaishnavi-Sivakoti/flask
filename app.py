from flask import Flask, flash, request, redirect, send_from_directory, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from cnnRnn import generate_image_description
from my_clip import generate_caption
from combining import generate_new_sentence
from videopre import extract_frames
from gtts import gTTS 
from blue import calculate_bleu_scores
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def generate_audio(combined_caption):
    # Generate audio file from the combined sentence
    tts = gTTS(text=combined_caption, lang='en')  # Language can be changed if needed
    audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_audio.mp3')
    tts.save(audio_file_path)
    return audio_file_path
    
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        extension = filename[-3:].lower()
        if extension=='mp4':
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
            file.save(video_path)

            # Extract frames from the video and generate captions
            extracted_data = extract_frames(video_path)
            return render_template('video.html', frames= extracted_data)
        else:
            image_path=os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(image_path)
            #print('upload_image filename: ' + filename)
            # Generate captions for the image
            caption1 = generate_caption(image_path)
            caption2 = generate_image_description(image_path)
            result = generate_new_sentence(caption1, caption2)
            audio_file_path = generate_audio(result)
            # res= calculate_bleu_scores(filename,caption1, caption2, result)
            # print(res[0]," ",res[1]," ",res[2])
            flash('Image successfully uploaded and displayed below')
            return render_template('index.html', filename='uploaded_image.jpg', caption1=caption1, caption2=caption2, combined_caption=result,audio_file='generated_audio.mp3')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    extension = filename[-3:].lower()
    print(filename)    
    if extension == 'mp4':
        # For video files, redirect to the frames directory

        return redirect(url_for('static', filename='uploads/frames101'+filename)[1:], code=301)
    else:
        # For image files, redirect to the uploaded image
        return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
@app.route('/display_audio/<filename>')
def display_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()