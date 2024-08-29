import argparse
import os

def load_filepaths_and_text(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            yield line 

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
        f.writelines([args.data_root_path+"|".join(line.split("|")[:2])+"\n" for line in load_filepaths_and_text(filelist)])