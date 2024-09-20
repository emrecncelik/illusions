import numpy as np
from scipy.io import wavfile
from vad import EnergyVAD


def read_wav(audio_file: str):
    return wavfile.read(audio_file)


def write_wav(audio: np.ndarray, sample_rate: int, output_file: str):
    wavfile.write(output_file, sample_rate, audio)


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
