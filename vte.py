import os
import torch
from datasets import Dataset, Audio
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2BertForCTC,
    Wav2Vec2ConformerForCTC,
    WhisperForConditionalGeneration,
)


# "openai/whisper-tiny.en" --- No effect
# "hf-audio/wav2vec2-bert-CV16-en" --- No effect
# "facebook/wav2vec2-base-960h" --- No effect


def load_model(model_name: str = "facebook/wav2vec2-base-960h"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    return processor, model


def get_audio_filenames(audio_dir: str):
    filenames = os.listdir(audio_dir)
    filenames = [os.path.join(audio_dir, f) for f in filenames if f.endswith(".wav")]
    return filenames


def load_dataset(filenames: list[str], sampling_rate: int = 16000):
    words = []
    repetitions = []
    for f in filenames:
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


def calculcate_unique_forms(transcription: str):
    unique_forms = list(dict.fromkeys(transcription.split()))
    return unique_forms


if __name__ == "__main__":
    single_words = get_audio_filenames("data/vte/single")
    repetitions = get_audio_filenames("data/vte/repetitions")

    processor, model = load_model()
    dataset = load_dataset(single_words + repetitions)

    unique_forms = []
    transcriptions = []
    for i in range(len(dataset)):
        print(f"Processing {i+1}/{len(dataset)}")
        print(f"\tWord: {dataset[i]['word']}")
        print(f"\tRep.: {dataset[i]['repetition']}")

        inputs = processor(
            dataset[i]["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.batch_decode(predicted_ids)[0]
        unique_forms.append(calculcate_unique_forms(transcription))
        transcriptions.append(transcription)
        print(f"\tUnique forms: {unique_forms[-1]}")

    unique_forms = ["|".join(form) for form in unique_forms]
    dataset = dataset.add_column("transcription", transcriptions)
    dataset = dataset.add_column("unique_forms", unique_forms)
    dataset = dataset.remove_columns(["audio"])

    dataset.to_csv("data/vte/transcriptions.csv")
