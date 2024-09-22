import os
import torch
import argparse
import numpy as np
from datasets import Dataset, Audio
from experiment_config import get_model_type_by_value, PROJECT_DIR
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    WhisperForConditionalGeneration,
    SpeechT5ForSpeechToText,
)

parser = argparse.ArgumentParser(
    description="Verbal transformation effect IO experiments."
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=os.path.join(PROJECT_DIR, "data/vte/warren1968"),
    help=(
        "Path to the data directory containing folders single and repetitions."
        "Folders should contain audio files."
    ),
)
parser.add_argument(
    "--model_name",
    type=str,
    default="facebook/wav2vec2-base-960h",
    help="Name of the model",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join(PROJECT_DIR, "results"),
    help="Path to the output directory",
)

args = parser.parse_args()


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


def get_audio_filenames(audio_dir: str):
    filenames = os.listdir(audio_dir)
    filenames = [os.path.join(audio_dir, f) for f in filenames if f.endswith(".wav")]
    return filenames


def load_repetition_dataset(filenames: list[str], sampling_rate: int = 16000):
    words = []
    repetitions = []
    for f in filenames:
        f = f.split("/")[-1]
        if any(filter(str.isdigit, f)):
            word = f.split("_")[0].split("/")[-1]
            repetition = f.split("_")[1].split(".")[0]
        else:
            word = f.split(".")[0].split("/")[-1]
            repetition = 1

        words.append(word)
        repetitions.append(int(repetition))

    return Dataset.from_dict(
        {"audio": filenames, "word": words, "repetition": repetitions}
    ).cast_column("audio", Audio(sampling_rate=sampling_rate))


def transcribe(
    model: AutoModelForCTC | WhisperForConditionalGeneration,
    model_type: str,
    processor: AutoProcessor,
    audio: np.ndarray,
):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
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


def calculcate_unique_forms(transcription: str):
    unique_forms = list(dict.fromkeys(transcription.split()))
    return unique_forms


if __name__ == "__main__":
    single = get_audio_filenames(os.path.join(args.data_dir, "single"))
    repetitions = get_audio_filenames(os.path.join(args.data_dir, "repetitions"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = get_model_type_by_value(args.model_name)
    model, processor = load_model(args.model_name, model_type)
    model = model.to(device)

    dataset = load_repetition_dataset(single + repetitions)

    unique_forms = []
    transcriptions = []
    transcription_lengths = []
    for i in range(len(dataset)):
        print(f"Processing {i+1}/{len(dataset)}")
        print(f"\tWord: {dataset[i]['word']}")
        print(f"\tRep.: {dataset[i]['repetition']}")

        transcription = transcribe(
            model, model_type, processor, dataset[i]["audio"]["array"]
        )

        unique_forms.append(calculcate_unique_forms(transcription))
        transcription_lengths.append(len(transcription.split()))
        transcriptions.append(transcription)
        print(f"\tTranscription: {transcription}")
        print(f"\tUnique forms: {unique_forms[-1]}")

    unique_forms = ["|".join(form) for form in unique_forms]
    dataset = dataset.add_column("transcription", transcriptions)
    dataset = dataset.add_column("unique_forms", unique_forms)
    dataset = dataset.add_column("transcription_length", transcription_lengths)
    dataset = dataset.remove_columns(["audio"])
    dataset.to_csv(os.path.join(args.output_dir, "transcriptions.csv"))
