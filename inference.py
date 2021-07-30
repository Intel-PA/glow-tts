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

GPU = True


device = torch.device('cuda' if GPU else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def get_output_filename(prefix, index):
  return f"./generated_wavs/{prefix}_speech_{index}.wav"

# get model_name
m_name = sys.argv[1]

# load WaveGlow
waveglow_path = '../glow_tts_pretrained_models/waveglow_256channels_ljs_v3.pt' # or change to the latest version of the pretrained WaveGlow.
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
_ = waveglow.to(device).eval()

# If you are using your own trained model 
#model_dir = "./logs/augmented_model_eighth/"
#hps = utils.get_hparams_from_dir(model_dir)
#checkpoint_path = utils.latest_checkpoint_path(model_dir) 
model_dir = f"./train_logs/{m_name}_run_0/"
hps = utils.get_hparams_from_dir(model_dir)
checkpoint_path = utils.latest_checkpoint_path(model_dir) 

# If you are using a provided pretrained model
hps = utils.get_hparams_from_file("./configs/base.json")
checkpoint_path = "../glow_tts_pretrained_models/glow_tts_pretrained.pt"

model = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to(device)

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

# normalizing & type casting
def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
    return np.clip((x / np.abs(x).max()) * max_wav_value, -32768, 32767).astype("int16")

strings = ["What kind of symptoms are you experiencing ?", "Do you have a high temperature ?", "The enormity of this task sometimes makes me feel a little dizzy but as a scientist and an explorer I have a duty to bear witness to the splendours of the world .", "I have observed that while the statues of a particular hall are more or less uniform in size there is considerable variation in halls .", "In some places the figures are three times the height of a human being in others life sized and in yet others only reach as high as my shoulder ."]
# tst_stn = "This was trained for one hundred and twelve epochs with no data augmentation."

from scipy.io.wavfile import write

for index, tst_stn in enumerate(strings):
    if getattr(hps.data, "add_blank", False):
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
        text_norm = commons.intersperse(text_norm, len(symbols))
    else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
        tst_stn = " " + tst_stn.strip() + " "
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
    sequence = np.array(text_norm)[None, :]
    print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)

    with torch.no_grad():
        noise_scale = .667
        length_scale = 1.0
        (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
        try:
            audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
        except:
            audio = waveglow.infer(y_gen_tst, sigma=.666)


    audio = normalize_audio(audio[0].clamp(-1,1).data.cpu().float().numpy())
 
    outfile = get_output_filename(m_name, index) 
    write(outfile, hps.data.sampling_rate, audio)
