import os

PROJECT_DIR = os.path.abspath(os.curdir)
STIMULI = {
    "warren1968": [
        "tress",
        "lame duck",
        "see",
        "trek",
        "seeshaw",
        "trice",
        "fill-up",
        "truce",
        "our ship has sailed",
        "ripe",
        "rape",
    ],
}

MODELS = {
    "wav2vec2": [
        "facebook/wav2vec2-base-100h",
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h",
        "facebook/wav2vec2-large-960h-lv60-self",
    ],
    "wav2vec2bert": ["hf-audio/wav2vec2-bert-CV16-en"],
    "speecht5": ["microsoft/speecht5_asr"],
    "whisper": ["openai/whisper-tiny.en"],
}
