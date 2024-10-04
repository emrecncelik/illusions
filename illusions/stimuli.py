import torch
import transformers
import numpy as np
from transformers import pipeline
from datasets import load_dataset
from vad import EnergyVAD


class Noise:
    @staticmethod
    def gaussian(
        audio: np.ndarray,
        noise_level: float = 0.1,
        mu: float = 0,
        sigma: float = 1,
    ) -> np.ndarray:
        noise = np.random.normal(mu, sigma, audio.shape)
        return audio + noise * noise_level

    @staticmethod
    def uniform(
        audio: np.ndarray,
        noise_level: float = 0.1,
        low: float = -1,
        high: float = 1,
    ) -> np.ndarray:
        noise = np.random.uniform(low, high, audio.shape)
        return audio + noise * noise_level

    @staticmethod
    def poisson(
        audio: np.ndarray,
        noise_level: float = 0.1,
        lam: float = 1,
    ) -> np.ndarray:
        noise = np.random.poisson(lam, audio.shape)
        return audio + noise * noise_level


def load_syntesizer(
    model_name: str = "microsoft/speecht5_tts",
    speaker_embed_idx: int = 150,
    device: str = "cpu",
) -> tuple[transformers.Pipeline, torch.Tensor]:
    """
    Load a text-to-speech synthesizer and speaker embedding.
    Args:
        model_name (str, optional): The name of the TTS model to use. Defaults to "microsoft/speecht5_tts".
        speaker_embed_idx (int, optional): The index of the speaker embedding to use from Matthijs/cmu-arctic-xvectors dataset. Defaults to 150.
        device (str, optional): The device to run the model on. Defaults to "cpu".
    Returns:
        Tuple[pipeline, torch.Tensor]: A tuple containing the synthesizer pipeline and the speaker embedding tensor.
    """
    synthesizer = pipeline("text-to-speech", model_name, device=device)
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(
        embeddings_dataset[speaker_embed_idx]["xvector"]
    ).unsqueeze(0)
    return synthesizer, speaker_embedding


def text2speech(
    text: str, synthesizer: transformers.Pipeline, speaker_embedding: torch.Tensor
):
    """Obvious I think"""
    speech = synthesizer(text, forward_params={"speaker_embeddings": speaker_embedding})
    return speech


def silent_gap(duration: int = 10, sampling_rate: int = 16000):
    """Creates a silent audio array of a given duration.

    Args:
        duration (int, optional): Duration of the audio in milliseconds. Defaults to 10.
        sampling_rate (int, optional): Sampling rate for the silence to be created. Defaults to 16000.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate and the silent audio array.
    """
    num_samples = int(sampling_rate * (duration / 1000.0))
    return sampling_rate, np.zeros(num_samples, dtype=np.float32)


def concatenate_audio(
    audio: tuple[int, np.ndarray], N: int, delimiter: tuple[int, np.ndarray] = None
):
    """
    Concatenates the audio from the input file N times.
    Adds a delimiter between each repetition if provided.
    Args:
        audio (str): The path to the input audio file.
        N (int): The number of times to repeat the audio.
        delimiter (tuple[int, np.ndarray], optional): A tuple delimiter sample rate and audio array. Defaults to None.
    Returns:
        tuple[np.ndarray, int]: A tuple containing the concatenated audio array and the sample rate.
    """
    if delimiter is None:
        audio_concat = np.concatenate([audio[1] for _ in range(N)])
    else:
        audio_concat = np.concatenate(
            [np.concatenate((audio[1], delimiter[1])) for _ in range(N)]
        )
    return audio[0], audio_concat


def remove_silent_edges(audio: np.ndarray, **kwargs):
    """
    If given audio has silent parts at the beginning and end, this function removes them.
    Args:
        audio (np.ndarray): The input audio waveform.
        **kwargs: Additional keyword arguments to be passed to EnergyVAD.
    Returns:
        np.ndarray: The new waveform with silent edges removed.
    """
    vad = EnergyVAD(**kwargs)
    voice_activity = vad(audio)
    indices = np.where(voice_activity == 1)[0]
    start = indices[0]
    end = indices[-1]

    voice_activity[start:end] = 1
    new_waveform = []
    for i in range(len(voice_activity)):
        if voice_activity[i] == 1:
            new_waveform.append(audio[i * 320 : (i + 1) * 320])
    new_waveform = np.concatenate(new_waveform)
    return new_waveform
