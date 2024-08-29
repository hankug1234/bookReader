from kserve import Model, ModelServer
from typing import Dict
import os,sys
sys.path.append(".")

import torch, commons, utils, glob, bson
import numpy as np 

from models import SynthesizerTrn
from text.symbols import symbols
from non_symbol_remover import symbols_filter
from scipy.io.wavfile import write
from text import cleaners

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

def bson_parsing(data):
    data = bson.loads(data)
    return data

class RvcModel(Model):
    def __init__(self,name:str):
        super().__init__(name)
        self.name = name
        self.model_path = os.environ.get("MODEL_PATH", "")
        self.load()
    
    def load(self):
        self.ready = True
        print(f'model_path : {os.environ.get("MODEL_PATH", "")}')
        self.hps = f'{self.model_path}/{os.path.basename(os.environ.get("MODEL_PATH", ""))}.json'
        self.checkpoint = list(filter(lambda x :True if 'G_' in x and '.pth' in x  else False, glob.glob(f'{self.model_path}/*')))[0]
        
        self.hps = utils.get_hparams_from_file(self.hps)
  
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        
        _ = self.net_g.eval()

        _ = utils.load_checkpoint(self.checkpoint, self.net_g, None)
        
        
    
    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        
        if os.path.exists(self.model_path) != True:
            return {"prediction" : "model not found", "code":404}
        
        if headers['content-type'] == 'application/x-www-form-urlencoded':
            datas = bson_parsing(payload)
        else:
            return {"prediction" : "must content-type = application/x-www-form-urlencoded", "code":500}
        
        
        if len(datas["scripts"]) <= 0 :
            return {"prediction" : f"scripts not found", "code":404}
        
        conversion_results = []
        
        try:
            for script in datas["scripts"]:
                p_script = get_text(script.strip('\n'),self.hps)
                with torch.no_grad():
                    x_tst = p_script.cuda().unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([p_script.size(0)]).cuda()
                    audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        
                    conversion_results.append({"sr":self.hps.data.sampling_rate,"audio":audio.tolist(),"dtype":str(audio.dtype)})
                
            return bson.dumps({"prediction" : conversion_results , "log":"", "code":200})
        except Exception as e:
            return {"prediction" : f"{e}", "code":404}
        

if __name__ == '__main__':
    model = RvcModel("tts")
    ModelServer().start([model])