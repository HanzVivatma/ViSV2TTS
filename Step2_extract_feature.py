import os, sys
import glob
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import glob, tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = VoiceEncoder(device=device)

def extract_speaker_embedding():
     wavs = glob.glob("DATA/wavs/*.wav")

     os.makedirs("DATA/embedding", exist_ok=True)
     for path in tqdm.tqdm(wavs):
          wav = preprocess_wav(path)
          embed = encoder.embed_utterance(wav)
          # print(embed.shape) # (256,)
          np.save(path.replace("wavs", "embedding").replace(".wav",".npy"), embed)

if __name__ == '__main__':
    extract_speaker_embedding()
