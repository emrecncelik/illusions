import os
import torch
import argparse
from scipy.io import wavfile
from datasets import load_dataset
from transformers import pipeline, set_seed

from illusions.config import STIMULI, PROJECT_DIR
from illusions.stimuli import (
    concatenate_audio,
    silent_gap,
    remove_silent_edges,
)


parser = argparse.ArgumentParser(
    description="Creates repeating stimuli for verbal transformation effect experiments."
)
parser.add_argument(
    "--synthesizer",
    type=str,
    default="microsoft/speecht5_tts",
    help="The name of the synthesizer model.",
)
parser.add_argument(
    "--speaker_embed_idx",
    type=int,
    default=150,
    help="The index of the speaker embedding",
)
parser.add_argument(
    "--stimuli",
    type=str,
    default="warren1968",
    help="The name of the stimuli set.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join(PROJECT_DIR, "data/vte/"),
    help="The output directory",
)
parser.add_argument(
    "--repetitions",
    nargs="+",
    type=int,
    default=None,
    help="The repetition values",
)
parser.add_argument(
    "--gap", type=float, default=None, help="The silent gap duration in milliseconds."
)
parser.add_argument(
    "--remove_silent_edges",
    default=False,
    action="store_true",
    help="Whether to remove silent edges from the audio",
)

args = parser.parse_args()

if __name__ == "__main__":
    set_seed(7)
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the synthesiser and the speaker embedding
    synthesiser = pipeline("text-to-speech", args.synthesizer, device=device)
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(
        embeddings_dataset[args.speaker_embed_idx]["xvector"]
    ).unsqueeze(0)

    # Create directories for the stimuli
    STIMULI_DIR = os.path.join(args.output_dir, args.stimuli)
    SINGLE_DIR = os.path.join(STIMULI_DIR, "single")

    if args.repetitions:
        REPETITIONS_DIR = os.path.join(STIMULI_DIR, "repetitions")

    os.makedirs(STIMULI_DIR, exist_ok=True)
    os.makedirs(SINGLE_DIR, exist_ok=True)
    if args.repetitions:
        os.makedirs(REPETITIONS_DIR, exist_ok=True)

    # Create the single word stimuli
    for stimulus in STIMULI[args.stimuli]:
        stimulus = stimulus.replace(" ", "-")
        print(f"Creating stimulus for {stimulus}")
        speech = synthesiser(
            stimulus, forward_params={"speaker_embeddings": speaker_embedding}
        )

        if args.remove_silent_edges:
            speech["audio"] = remove_silent_edges(
                speech["audio"],
                energy_threshold=0.001,  # better val. to catch consonant endings
            )

        wavfile.write(
            os.path.join(SINGLE_DIR, f"{stimulus}.wav"),
            speech["sampling_rate"],
            speech["audio"],
        )

    if args.repetitions:
        # Create the repetition stimuli
        audio_files = os.listdir(SINGLE_DIR)
        audio_files = [os.path.join(SINGLE_DIR, file) for file in audio_files]

        if args.gap is not None:
            silence = silent_gap(args.gap)

        for rep in args.repetitions:
            for audio_file in audio_files:
                print(f"Creating repetition {rep} for {audio_file}")
                audio = wavfile.read(audio_file)
                stimulus_name = os.path.basename(audio_file).split(".")[0]
                filename = (
                    f"{stimulus_name}_{rep}_{int(args.gap)}.wav"
                    if args.gap
                    else f"{stimulus_name}_{rep}.wav"
                )
                output_file = os.path.join(REPETITIONS_DIR, filename)

                if args.gap is None:
                    audio = concatenate_audio(audio, rep)
                else:
                    audio = concatenate_audio(audio, rep, silence)

                wavfile.write(output_file, audio[0], audio[1])
