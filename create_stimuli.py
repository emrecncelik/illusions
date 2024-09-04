import os
import torch
from transformers import pipeline, set_seed, logging
from datasets import load_dataset
from scipy.io import wavfile
from stimuli import WORDS
from audio_utils import concatenate_audio, silent_gap, read_wav, write_wav


SYNTHESIZER = "microsoft/speecht5_tts"
SPEAKER_EMBED_IDX = 200
STIMULI = "warren1961"
OUTPUT_DIR = "./data/vte/"
DIR_SINGLE = os.path.join(OUTPUT_DIR, "single")
DIR_REPETITIONS = os.path.join(OUTPUT_DIR, "repetitions")
REPETITIONS = (100, 200, 300)
GAP = None


if __name__ == "__main__":
    set_seed(7)
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the synthesiser and the speaker embedding
    synthesiser = pipeline("text-to-speech", SYNTHESIZER, device=device)
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(
        embeddings_dataset[SPEAKER_EMBED_IDX]["xvector"]
    ).unsqueeze(0)

    # Create directories for the stimuli
    os.makedirs(DIR_SINGLE, exist_ok=True)
    os.makedirs(DIR_REPETITIONS, exist_ok=True)

    # Create the single word stimuli
    for word in WORDS["warren1961"]:
        print(f"Creating stimulus for {word}")
        speech = synthesiser(
            word, forward_params={"speaker_embeddings": speaker_embedding}
        )
        write_wav(
            speech["audio"],
            speech["sampling_rate"],
            os.path.join(DIR_SINGLE, f"{word}.wav"),
        )

    # Create the repetition stimuli
    audio_files = os.listdir(DIR_SINGLE)
    audio_files = [os.path.join(DIR_SINGLE, file) for file in audio_files]

    for rep in REPETITIONS:
        for audio_file in audio_files:
            print(f"Creating repetition {rep} for {audio_file}")
            audio = read_wav(audio_file)
            stimulus_name = os.path.basename(audio_file).split(".")[0]
            output_file = os.path.join(DIR_REPETITIONS, f"{stimulus_name}_{rep}.wav")

            if GAP is None:
                concatenate_audio(audio, rep)
            else:
                concatenate_audio(audio, rep, silent_gap(GAP, audio[0]))

            write_wav(audio[1], audio[0], output_file)
