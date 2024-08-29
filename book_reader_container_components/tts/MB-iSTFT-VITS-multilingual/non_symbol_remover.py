import sys 
sys.path.append("..")
from text.symbols import symbols
import argparse
import os,re 

def load_filepaths_and_text(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            yield line 

def symbols_filter(text):
  return re.sub(f'[^{"".join(symbols)}]'," ",text)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--data_root_path",default="./datas/kss/")
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])

  args = parser.parse_args()
  
  for filelist in args.filelists:
    print("START:", filelist)
    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
        result = []
        for line in load_filepaths_and_text(filelist):
          tokens = line.split("|")
          filtered_text = symbols_filter(tokens[1])
          result.append(tokens[0] + "|" + filtered_text +'\n')
        f.writelines(result)          
