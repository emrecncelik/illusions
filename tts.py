import os
import torch
import numpy as np
from transformers import pipeline
from datasets import load_dataset
from scipy.io import wavfile
from stimuli import WORDS


if __name__ == "__main__":
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    for word in WORDS["warren1961"]:
        speech = synthesiser(
            word, forward_params={"speaker_embeddings": speaker_embedding}
        )
        wavfile.write(
            f"data/vte_words/__{word}.wav", speech["sampling_rate"], speech["audio"]
        )

    data_dir = "data/vte/"
    audio_files = os.listdir(os.path.join(data_dir, "words"))
    audio_files = [os.path.join(data_dir, "words", file) for file in audio_files]

    for rep in (100, 200, 300):
        for audio_file in audio_files:
            audio = wavfile.read(audio_file)[1]
            word = os.path.basename(audio_file).split(".")[0]
            output_file = f"data/vte/repetitions/{word}_{rep}.wav"
            concatenate_audio(audio_file, output_file, rep)
