from datasets import load_dataset
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import whisper
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
import torch
import soundfile as sf
import os
from tempfile import NamedTemporaryFile
from io import BytesIO

app = FastAPI()

# Ignore SSL certificate errors (for loading models) Just for testing purposes
## Comment the following 2 lines in production
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Serve static files (the HTML page)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models (same as before)
whisper_model = whisper.load_model("small")
mbart_model_name = 'facebook/mbart-large-50-many-to-many-mmt'
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_name)
mbart_tokenizer.src_lang = "fr_XX"
mbart_tokenizer.tgt_lang = "en_XX"
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Helper function for generating audio from text (same as before)
def generate_audio(input_text: str, out_dir: str) -> None:
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    inputs = processor(text=input_text, return_tensors="pt", truncation=True, padding="max_length")
    set_seed(555)
    speech = speech_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write(out_dir, speech.numpy(), samplerate=16000)

# Process audio upload and response
@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    temp_audio_file = NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        contents = await file.read()
        temp_audio_file.write(contents)
        temp_audio_file.close()

        result = whisper_model.transcribe(temp_audio_file.name)
        transcription = result['text']

        inputs = mbart_tokenizer(transcription, return_tensors="pt")
        translated = mbart_model.generate(**inputs)
        translated_text = mbart_tokenizer.decode(translated[0], skip_special_tokens=True)

        output_audio_path = "./output_audio.wav"
        generate_audio(translated_text, output_audio_path)

        return FileResponse(output_audio_path, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output_audio.wav"})

    finally:
        if os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)

# Serve the UI
@app.get("/")
async def root():
    return FileResponse("static/index.html")