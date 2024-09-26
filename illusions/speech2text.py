import torch
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    WhisperForConditionalGeneration,
    SpeechT5ForSpeechToText,
)


def load_model(model_name: str = "facebook/wav2vec2-base-960h", model_type: str = None):
    processor = AutoProcessor.from_pretrained(model_name)

    if model_type in ("wav2vec2", "wav2vec2bert", "vawlm"):
        model = AutoModelForCTC.from_pretrained(model_name)
    elif model_type == "speecht5":
        model = SpeechT5ForSpeechToText.from_pretrained(model_name)
    elif model_type == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    return model, processor


def transcribe(
    model: AutoModelForCTC | WhisperForConditionalGeneration,
    model_type: str,
    processor: AutoProcessor,
    audio: np.ndarray,
    device: torch.device,
):

    inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if model_type in ("wav2vec2", "wav2vec2bert", "wavlm"):
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    elif model_type == "speecht5":
        predicted_ids = model.generate(**inputs, max_length=100)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]
    elif model_type == "whisper":
        raise NotImplementedError("Whisper model not implemented.")
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    return transcription
