import sys
sys.path.append('./waveglow/')
import IPython.display as ipd

import librosa
import numpy as np
import os
import glob
import json

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils


outfile = "./generated_wavs/test.wav"


# load WaveGlow
waveglow_path = '../glow_tts_pretrained_models/waveglow_256channels_ljs_v3.pt' # or change to the latest version of the pretrained WaveGlow.
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
_ = waveglow.cuda().eval()

# If you are using your own trained model
model_dir = "./logs/base/"
hps = utils.get_hparams_from_dir(model_dir)
checkpoint_path = utils.latest_checkpoint_path(model_dir)

# If you are using a provided pretrained model
#hps = utils.get_hparams_from_file("./configs/base.json")
#checkpoint_path = "../glow_tts_pretrained_models/glow_tts_pretrained.pt"

model = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to("cuda")

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

# normalizing & type casting
def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
    return np.clip((x / np.abs(x).max()) * max_wav_value, -32768, 32767).astype("int16")


tst_stn = "This was trained for one hundred and twelve epochs with no data augmentation." 

if getattr(hps.data, "add_blank", False):
    text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
    text_norm = commons.intersperse(text_norm, len(symbols))
else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
    tst_stn = " " + tst_stn.strip() + " "
    text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
sequence = np.array(text_norm)[None, :]
print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

with torch.no_grad():
  noise_scale = .667
  length_scale = 1.0
  (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
  try:
    audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
  except:
    audio = waveglow.infer(y_gen_tst, sigma=.666)


audio = normalize_audio(audio[0].clamp(-1,1).data.cpu().float().numpy())

from scipy.io.wavfile import write

write(outfile, hps.data.sampling_rate, audio)
