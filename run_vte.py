import os
import torch
import argparse
from datasets import Dataset, Audio
from illusions.config import get_model_type_by_value, PROJECT_DIR
from illusions.speech2text import load_model, transcribe

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


def get_audio_filenames(audio_dir: str):
    filenames = os.listdir(audio_dir)
    filenames = [os.path.join(audio_dir, f) for f in filenames if f.endswith(".wav")]
    return filenames


def load_repetition_dataset(filenames: list[str], sampling_rate: int = 16000):
    words = []
    repetitions = []
    gaps = []
    for f in filenames:
        f = f.split("/")[-1]
        if any(filter(str.isdigit, f)):
            word = f.split("_")[0].split("/")[-1]

            if f.count("_") > 1:
                repetition = f.split("_")[-2]
                gap = f.split("_")[-1].split(".")[0]
            else:
                repetition = f.split("_")[1].split(".")[0]
                gap = None
        else:
            word = f.split(".")[0].split("/")[-1]
            repetition = 1
            gap = None

        gaps.append(gap)
        words.append(word)
        repetitions.append(int(repetition))

    return Dataset.from_dict(
        {"audio": filenames, "word": words, "repetition": repetitions, "gap": gaps}
    ).cast_column("audio", Audio(sampling_rate=sampling_rate))


def calculcate_unique_forms(transcription: str):
    unique_forms = list(dict.fromkeys(transcription.split()))
    return unique_forms


def calculate_transitions(transcription: str):
    num_transitions = 0
    loc_transitions = []
    words = transcription.split()
    for i in range(len(words) - 1):
        if words[i] != words[i + 1]:
            num_transitions += 1
            loc_transitions.append(i)
    return num_transitions, loc_transitions


if __name__ == "__main__":
    single = get_audio_filenames(os.path.join(args.data_dir, "single"))
    repetitions = get_audio_filenames(os.path.join(args.data_dir, "repetitions"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = get_model_type_by_value(args.model_name)
    model, processor = load_model(args.model_name, model_type)
    model = model.to(device)

    dataset = load_repetition_dataset(single + repetitions)

    unique_forms = []
    transitions = []
    transcriptions = []
    transcription_lengths = []
    for i in range(len(dataset)):
        print(f"Processing {i+1}/{len(dataset)}")
        print(f"\tWord: {dataset[i]['word']}")
        print(f"\tRep.: {dataset[i]['repetition']}")

        transcription = transcribe(
            model,
            model_type,
            processor,
            dataset[i]["audio"]["array"],
            device,
        )

        unique_forms.append(calculcate_unique_forms(transcription))
        transitions.append(calculate_transitions(transcription))
        transcription_lengths.append(len(transcription.split()))
        transcriptions.append(transcription)
        print(f"\tTranscription: {transcription}")
        print(f"\tUnique forms: {unique_forms[-1]}")
        print(f"\tNum. transitions: {transitions[-1][0]}")
        print(f"\tLoc. of transitions: {transitions[-1][1]}")

    unique_forms = ["|".join(form) for form in unique_forms]
    dataset = dataset.add_column("transcription", transcriptions)
    dataset = dataset.add_column("unique_forms", unique_forms)
    dataset = dataset.add_column("transcription_length", transcription_lengths)
    dataset = dataset.remove_columns(["audio"])
    dataset.to_csv(os.path.join(args.output_dir, "transcriptions.csv"))
