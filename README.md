# Illusions
This repository contains the experiments for my master's thesis \<TITLE HERE\>. We explore the dynamics of auditory and audio-visual illusions using artificial neural networks. Current focus is on auditory illusions however we aim to also include AV speech illusions such as McGurk effect.
## Auditory


### Verbal Transformation Effect

For initial experiments, we use the same repeating stimuli from Warren (1968) and a set of pre-trained speech-to-text models: Wav2Vec2 and Wav2Vec2BERT with connectionist temporal classification, Whisper with conditional generation. 

Conventional speech-to-text models do not show the verbal transformation effect (as far as the simple experiments go). This is in line with the spreading activations and habituation hypotheses for verbal transformations. Maybe try out [continous integrate and fire](https://github.com/MingLunHan/CIF-PyTorch) models which are more biologically plausible and may show effects like habituation.

**TODO:**
- [x] Wav2Vec2ForCTC - No effect observed
- [x] Wav2Vec2BertForCTC - No effect observed
- [ ] Wav2Vec2ConformerForCTC
- [x] WhisperForConditionalGeneration - No effect observed


# References
Warren, R. M. (1968). Verbal transformation effect and auditory perceptual mechanisms. Psychological Bulletin, 70(4), 261–270. https://doi.org/10.1037/h0026275