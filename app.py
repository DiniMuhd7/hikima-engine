import os
import torch
import random
import docx
import PyPDF2
import json

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import requests
import uuid

from flask import (Flask, redirect, render_template, request, session,
                   send_from_directory, url_for, jsonify)

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

app = Flask(__name__)
app.secret_key = '\x10\x95M\xc4u\xed\x0c\xb7\xaa\x90:\xfc\xe9\x9c\x98\xbc\xa2d\x7f-/\x08*\x86'  # Set a secret key for session encryption.


# Initialize Google Drive API
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

account_sid = 'AC65795a522b5006f5fd8fa139647ac947'
auth_token = '1dce9d48c8505b7d6dba942a28bc5394'
client = Client(account_sid, auth_token)


# Define your Google Drive folder ID
GOOGLE_DRIVE_FOLDER_ID = 'hikima-data'

# Dictionary to store user recordings and associated metadata
user_recordings = {}


# Dictionary to store user credentials (replace with your user database)
users = {
    'demo@hikima.ai': 'pass-X202E',
    'admin@hikima.ai': 'pass-E202X',
}


@app.route('/webhook', methods=['POST'])
def webhook():
	try:
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body='Do you want to create a digital version of your voice? Send us your voice recordings, we will train an AI model that will sould like you.',
            to='whatsapp:+2348139476119'
        )
        if request.method == 'POST' and 'MediaUrl0' in request.form:
            media_url = request.form['MediaUrl0']
            user_email = request.form['From']
            language_type = request.form['Body']
            
            voice_recording_path = download_voice_recording(media_url)
            file_id = upload_to_google_drive(voice_recording_path)
            clean_up_temporary_file(voice_recording_path)
            
            save_user_recording(user_email, language_type, file_id)
            
            # Add code for model training integration
            
            response = MessagingResponse()
            response.message('Thank you for your voice recording!')
            return str(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred."


@app.route('/')
def index():
    if 'username' in session:
        #return f"Logged in as {session['username']} | <a href='/logout'>Logout</a>"
        return redirect(url_for('inference'))
    else:
        #return "Not logged in | <a href='/login'>Login</a>"
        return redirect(url_for('login'))



@app.route('/welcome')
def welcome():
    return render_template('index.html')



@app.route('/profile')
def profile():
    sentences = [
        "Bringing technology to your doorstep in an accelarating time.",
        "Under whose authority does a machine learning dataset is built?",
        "Overfitting is a major problem in training an AI model, in what ways can this be addressed?"
    ]
 
    return render_template('profile.html',sentences=sentences)


@app.route('/record', methods=['POST'])
def record():
    user_id = request.form.get('user_id')
    sentence = request.form.get('sentence')

    # Replace this with your implementation to handle the recording and saving logic
    # Save the recording to a directory named after the user ID
    audio_file = request.files['audio']
    if audio_file:
        directory = f"recordings/{user_id}"
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{sentence.replace(' ', '_')}.wav")
        audio_file.save(filepath)

        return jsonify(success=True, message="Recording saved successfully!")

    return jsonify(success=False, message="Failed to save the recording.")


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        message = "Access Denied. Contact @muhdsdeen7 for login details."

        if username != "" and password != "":
            if username in users and users[username] == password:
                session['username'] = username
                return redirect(url_for('inference'))
            else:
                return render_template('login.html', message=message)
    return render_template('login.html', message='')


def download_voice_recording(media_url):
    file_extension = media_url.split('.')[-1]
    voice_recording_path = f'temp/{str(uuid.uuid4())}.{file_extension}'
    response = requests.get(media_url, stream=True)
    
    with open(voice_recording_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    return voice_recording_path

def upload_to_google_drive(file_path):
    file = drive.CreateFile({'title': os.path.basename(file_path), 'parents': [{'id': GOOGLE_DRIVE_FOLDER_ID}]})
    file.SetContentFile(file_path)
    file.Upload()
    return file['id']

def clean_up_temporary_file(file_path):
    os.remove(file_path)

def save_user_recording(user_email, language_type, file_id):
    if user_email in user_recordings:
        user_recordings[user_email].append((language_type, file_id))
    else:
        user_recordings[user_email] = [(language_type, file_id)]


# Read texts
def read_texts(model_id, sentence, filename, device="cpu", language="", speaker_reference=None, faster_vocoder=False):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=faster_vocoder)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


# Setup model for inference experiment
def the_raven(version, model_id, exsentence, exec_device="cpu", speed_over_quality=False, speaker_reference=None, langcode=""):
    if langcode != "" and langcode == "ha":
        twicklang = "sw"
        read_texts(model_id="Hausa",
               sentence=exsentence,
               filename=f"static/audios/madugu_{version}.wav",
               device=exec_device,
               language=twicklang,
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
        #return render_template('inference.html', audio_visible=True, audio_file=audio_path)
    elif langcode != "" and langcode == "sw":
        read_texts(model_id="Swahili",
               sentence=exsentence,
               filename=f"static/audios/tai_{version}.wav",
               device=exec_device,
               language=langcode,
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)


# Initiate inference for experiment
@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if 'username' in session:
        #return f"Logged in as {session['username']} | <a href='/logout'>Logout</a>"
        #redirect(url_for('inference'))
        os.makedirs("static/audios", exist_ok=True)
        sentence = request.form.get('syn')
        langtag = request.form.get('tag')
        #print(str(langtag))
    
        digit = random.randint(0, 1000)
        exec_device = "cuda" if torch.cuda.is_available() else "cpu"
        message = "Listen to Speech"
        #messagex = f"Running on {exec_device}. Synthetizing..."
        if sentence != "" and langtag != "":
            if langtag != "" and langtag == "ha": 
                the_raven(version=digit,
                    model_id="Hausa",
                    exec_device=exec_device,
                    speed_over_quality=exec_device != "cuda",
                    exsentence=sentence,
                    langcode=langtag)
                audio_path = os.path.join('static/audios',f'madugu_{digit}.wav')
                return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
            elif langtag != "" and langtag == "sw":
                the_raven(version=digit,
                    model_id="Swahili",
                    exec_device=exec_device,
                    speed_over_quality=exec_device != "cuda",
               	    exsentence=sentence,
                    langcode=langtag)
                audio_path = os.path.join('static/audios',f'tai_{digit}.wav')
                return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)

        return render_template('inference.html', langtag = langtag, audio_visible=False, audio_file='', message='')
    else:
        #return "Not logged in | <a href='/login'>Login</a>"
        return redirect(url_for('login'))

# File extensions
def allowed_file(filename):
    # Check if the file extension is allowed
    allowed_extensions = {'txt', 'pdf', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def check_file_extension(file_path, extension):
    file_ext = os.path.splitext(file_path)[1]
    return file_ext.lower() == extension.lower()


# Read files
txt_archive = []
def read_txtfile(file_path):
    txt_archive.clear()
    with open(file_path, 'r') as file:
        for line in file:
            txt_archive.append(line.strip())
    return txt_archive

pdf_archive = []
def read_pdffile(file_path):
    pdf_archive.clear()
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        for page_number in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_number)
            text = page.extract_text()
            pdf_archive.append(text)

    return pdf_archive

doc_archive = []
def read_docfile(file_path):
    doc_archive.clear()
    doc = docx.Document(file_path)
    for paragraph in doc.paragraphs:
        text = paragraph.text
        doc_archive.append(text)
    
    return doc_archive

docx_archive = []
def read_docxfile(file_path):
    docx_archive.clear()
    doc = docx.Document(file_path)
    for paragraph in doc.paragraphs:
        text = paragraph.text
        docx_archive.append(text)
    
    return docx_archive


@app.route('/inferencex', methods=['GET', 'POST'])
def inferencex():
    if 'username' in session:
        #return f"Logged in as {session['username']} | <a href='/logout'>Logout</a>"
        #redirect(url_for('inference'))
        os.makedirs("static/audios", exist_ok=True)
        os.makedirs("static/uploads/docs", exist_ok=True)
        #sentence = request.form.get('syn')
        langtag = request.form.get('tagd')
        #print(str(langtag))

    
	    # Check if a file was uploaded
        if 'docfile' not in request.files:
            return 'No file uploaded.'
	
        file = request.files['docfile']

        # Check if the file exists and has an allowed extension
        if file.filename == '' or not allowed_file(file.filename):
            return 'Invalid file.'

        # Save the uploaded file
        file_path = os.path.join('static/uploads/docs', file.filename)
        file.save(file_path)
    
        txt_extension='.txt'
        pdf_extension='.pdf'
        doc_extension='.doc'
        docx_extension='.docx'
	
        is_matching_txt = check_file_extension(file_path, txt_extension)	
        is_matching_pdf = check_file_extension(file_path, pdf_extension)
        is_matching_doc = check_file_extension(file_path, doc_extension)
        is_matching_docx = check_file_extension(file_path, docx_extension)
	
        audio_path = ""
        if is_matching_txt is True:
            #print('Request: '+ str(is_matching_txt)+str(langtag))
            digit = random.randint(0, 1000)
            exec_device = "cuda" if torch.cuda.is_available() else "cpu"
            message = "Listen to Speech"
        
            sentences = read_txtfile(file_path)
            if sentences != "" and langtag != "":
                if langtag != "" and langtag == "ha":
                    the_raven(version=digit,
                        model_id="Hausa",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
                        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'madugu_{digit}.wav')
                    #audio_path = send_from_directory(os.path.join(app.root_path, 'audios'),f'madugu_{digit}.wav', mimetype='audio/wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
                elif langtag != "" and langtag == "sw": 
                    the_raven(version=digit,
                        model_id="Swahili",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
               	        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'tai_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
        elif is_matching_pdf is True:
	        #print('Request: '+ str(is_matching_pdf)+str(langtag))
            digit = random.randint(0, 1000)
            exec_device = "cuda" if torch.cuda.is_available() else "cpu"
            message = "Listen to Speech"
        
            sentences = read_pdffile(file_path)
            if sentences != "" and langtag != "":
                if langtag != "" and langtag == "ha":
                    the_raven(version=digit,
                        model_id="Hausa",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
                        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'madugu_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
                elif langtag != "" and langtag == "sw": 
                    the_raven(version=digit,
                        model_id="Swahili",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
               	        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'tai_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
        elif is_matching_doc is True:
	        #print('Request: '+ str(is_matching_doc)+str(langtag))
            digit = random.randint(0, 1000)
            exec_device = "cuda" if torch.cuda.is_available() else "cpu"
            message = "Listen to Speech"
        
            sentences = read_docfile(file_path)
            if sentences != "" and langtag != "":
                if langtag != "" and langtag == "ha":
                    the_raven(version=digit,
                        model_id="Hausa",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
                        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'madugu_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
                elif langtag != "" and langtag == "sw": 
                    the_raven(version=digit,
                        model_id="Swahili",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
               	        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'tai_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
        elif is_matching_docx is True:
	        #print('Request: '+ str(is_matching_docx)+str(langtag))	
            digit = random.randint(0, 1000)
            exec_device = "cuda" if torch.cuda.is_available() else "cpu"
            message = "Listen to Speech"
        
            sentences = read_docxfile(file_path)
            if sentences != "" and langtag != "":
                if langtag != "" and langtag == "ha":
                    the_raven(version=digit,
                        model_id="Hausa",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
                        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'madugu_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)
                elif langtag != "" and langtag == "sw": 
                    the_raven(version=digit,
                        model_id="Swahili",
                        exec_device=exec_device,
                        speed_over_quality=exec_device != "cuda",
               	        exsentence=sentences,
                        langcode=langtag)
                    audio_path = os.path.join('static/audios',f'tai_{digit}.wav')
                    return render_template('inference.html', langtag = langtag, audio_visible=True, audio_file=audio_path, message=message)

        #os.rmdir('static/audios')
        #os.rmdir('static/uploads/docs')
        return render_template('inference.html', langtag = langtag, audio_visible=False, audio_file='', message='')
    else:
        #return "Not logged in | <a href='/login'>Login</a>"
        return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run()
