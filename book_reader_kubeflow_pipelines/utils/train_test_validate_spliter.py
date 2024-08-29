from sklearn.model_selection import train_test_split
import argparse

def data_split(data_path,ratio):
    with open(data_path,"r",encoding="utf-8") as f:
        datas = f.readlines()
        train, rest,_, _ = train_test_split(datas,list(range(len(datas))),test_size=ratio,shuffle=True)
        test, validate,_,_ = train_test_split(rest,list(range(len(rest))),test_size=ratio,shuffle=True)
        with open(data_path+".train","w",encoding="utf-8") as f:
            f.writelines(train)
        with open(data_path+".test","w",encoding="utf-8") as f:
            f.writelines(test)
        with open(data_path+".validate","w",encoding="utf-8") as f:
            f.writelines(validate)
        
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root_path")
  parser.add_argument("--ratio",default=0.3, type=float)
  
  args = parser.parse_args()  
  data_split(args.data_root_path,args.ratio)
  