
import torch
import torchaudio
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = FastAPI()

processor = Wav2Vec2Processor.from_pretrained("C:\\Users\\hp\\Desktop\\ASR\\Finetuning\\modelsfolder",local_files_only=True)
model = Wav2Vec2ForCTC.from_pretrained("C:\\Users\\hp\\Desktop\\ASR\\Finetuning\\modelsfolder",local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def transcribe_audio_array(audio, sr):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read audio from uploaded file
        audio, sr = sf.read(file.file)
        transcription = transcribe_audio_array(audio, sr)
        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
#Finetuning/61-70968-0005.flac