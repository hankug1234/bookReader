import sys 
sys.path.append(".")
import argparse

import os
import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from non_symbol_remover import symbols_filter

from scipy.io.wavfile import write

from text import cleaners


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = symbols_filter(_clean_text(text, cleaner_names))
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--hps",default="../configs/kss.json")
  parser.add_argument("--checkpoint", default="../logs/kss/G_100000.pth")
  parser.add_argument("--filelists", nargs="+")
  parser.add_argument("--save_dir", default="/TTS/tts_result/kss")
  
  args = parser.parse_args()
  
  hps = utils.get_hparams_from_file(args.hps)
  
  net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
  
  _ = net_g.eval()

  _ = utils.load_checkpoint(args.checkpoint, net_g, None)
  
  os.makedirs(args.save_dir,exist_ok=True)
  
  for file in args.filelists:
    with open(file,"r",encoding="utf-8") as f:
      lines = f.readlines()
      for text in lines:
          p_text = get_text(text.strip('\n'),hps)
          with torch.no_grad():
            x_tst = p_text.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([p_text.size(0)]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            write(args.save_dir + os.sep + text + ".wav",hps.data.sampling_rate,audio)
