
# Speech Recognition System using Wav2Vec2 + FastAPI

This repository contains the complete pipeline for an Automatic Speech Recognition (ASR) system using a fine-tuned Wav2Vec2 model. It includes scripts for data loading, preprocessing, fine-tuning, evaluation, and serving the model via a FastAPI endpoint.



## Data Exploration,Evaluation and Finetuning
https://colab.research.google.com/drive/11E8rps0ObHdVwaHkEPYWCtsVqWus9lMS


## FastApi Service

### 1.Create and Activate Virtual Environment
```bash
  python -m venv venv
  venv\Scripts\activate
```

### 2.Install requirements
```bash
 pip install fastapi uvicorn transformers torchaudio soundfile

```

### 3.Model Download
https://drive.google.com/drive/folders/1MZdlmYTMGzUTm8tYYAEqryO_tPpWbGX5?usp=sharing

## 4.Run the fastapi
$ curl -v -X POST http://127.0.0.1:8000/transcribe/ \
>   -F "file=@filepath"




