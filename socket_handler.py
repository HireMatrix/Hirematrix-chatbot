import socketio
from vosk import Model, KaldiRecognizer
import json
import base64
import io
import aiohttp
import re
import fitz
from aiohttp import web

model = Model("models/vosk-model-en-us-0.42-gigaspeech")

sio = socketio.AsyncServer(
    cors_allowed_origins=["http://localhost:5173"],
    async_mode='aiohttp',
    max_http_buffer_size=100*1024*1024
)

app = web.Application()

recognizers = {}

def extract_text_from_pdf(file_stream):
    doc = fitz.open(stream=file_stream, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_extracted_text(text):
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = text.replace('ﬀ', 'ff').replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('"', "'")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def register_socket_events(sio):
    @sio.event
    async def connect(sid, environ):
        print(f"Connected: {sid}")
        recognizers[sid] = KaldiRecognizer(model, 16000)
        recognizers[sid].SetWords(True)

    @sio.event
    async def disconnect(sid):
        print(f"Disconnected: {sid}")
        if sid in recognizers:
            del recognizers[sid]

    @sio.event
    async def audio_chunk(sid, data):
        try:
            recognizer = recognizers.get(sid)
            if recognizer is None:
                return

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                await sio.emit("transcript", result.get("text", ""), to=sid)
            else:
                partial = json.loads(recognizer.PartialResult())
                await sio.emit("partial", partial.get("partial", ""), to=sid)

        except Exception as e:
            print(f"Error: {str(e)}")
            await sio.emit("error", f"Error processing audio: {str(e)}", to=sid)

    @sio.event
    async def analyze_resume(sid, resumeData):
        print(f"Resume received from {sid}")

        try:
            await sio.emit("resumeStatus", { "message": "Extracting Text..." }, to=sid)
            file_data = base64.b64decode(resumeData)
            file_stream = io.BytesIO(file_data)
            extracted_text = extract_text_from_pdf(file_stream)

            await sio.emit("resumeStatus", { "message": "Cleaning the Extracted Text..." }, to=sid)

            extracted_text = clean_extracted_text(extracted_text)

            await sio.emit("resumeStatus", { "message": "Analyzing Resume..."}, to=sid)

            async with aiohttp.ClientSession() as session:
                url = "http://127.0.0.1:8000/resumeAnalyzer"
                payload = {"prompt": extracted_text}

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        analyze_result = response_data
                    else:
                        analyze_result = f"Failed to analyze resume: {response.status}"

            await sio.emit("resumeResult", { "analyzed_result": analyze_result }, to=sid)

        except Exception as e:
            print(f"Error in analyzing resume: {str(e)}")
            await sio.emit("error", f"Error analyzing resume: {str(e)}", to=sid)
            

register_socket_events(sio)
sio.attach(app)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=5000)