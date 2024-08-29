from subprocess import Popen
from multiprocessing import Queue, Process
from time import sleep
import argparse, json


def if_done_multi(queue, cmds):
    done = True
    p = None
    try:
        while 1:
            flag = 1
            if done == True:
                queue.put(f"{cmds[0]} start")
                p = Popen(cmds[0], shell=True)
                done = False
                flag = 0
            else:
                if p is None or p.poll() is None:
                    flag = 0
                    sleep(0.5)
                else :
                    queue.put(f'{cmds[0]} done')
                    cmds.pop(0)
                    done = True
                    
                    if len(cmds) != 0:
                        flag = 0
                        
            if flag == 1:
                break
    except Exception as e:
        queue.put(e)
    finally:
        queue.put("done")
        queue.close()

def preprocesss_pipeline(cmds): 
    queue = Queue()
    
    process = Process(target=if_done_multi, args=(queue, cmds))
    process.start()
    
    while True:
        log = queue.get()
        if log == "done":
            yield log
            break
        yield log
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train",default='false',choices=['true','false'])
    parser.add_argument("--text_index",default=1,type=int)
    parser.add_argument("--filelists",type=str)
    parser.add_argument("--config",type=str)
    parser.add_argument("--text_cleaners",default="korean_cleaners",type=str)
    parser.add_argument("--opt_dir",type=str)
    parser.add_argument("--checkpoint",type=str)
  
    args = parser.parse_args()
    
    
    cmds = []
    
    if args.is_train == 'true' :
        text_cleaner_cmd = f"python /TTS/preprocess.py --out_extension cleaned --text_index {args.text_index} --filelists {args.filelists} --text_cleaners {args.text_cleaners}"
        symbol_filter_cmd = f"python /TTS/non_symbol_remover.py --out_extension filtered --filelists {args.filelists}.cleaned"
        test_validate_slit_cmd = f"python /TTS/train_test_validate_spliter.py --filelist {args.filelists}.cleaned.filtered"
        
        with open(args.config,"r",encoding="utf-8") as f:
            config_file = json.load(f)
            config_file["data"]["training_files"] = f"{args.filelists}.cleaned.filtered.train"
            config_file["data"]["validation_files"] = f"{args.filelists}.cleaned.filtered.validate"
            with open(args.config,"w",encoding="utf-8") as w:
                json.dump(config_file,w)
                        
        train_model_cmd = f"python /TTS/train_latest.py -c {args.config} -m {args.opt_dir}"
        cmds =[text_cleaner_cmd,symbol_filter_cmd ,test_validate_slit_cmd,train_model_cmd]
    else:
        tts_cmd = f"python /TTS/tts.py --hps {args.config} --checkpoint {args.checkpoint} --save_dir {args.opt_dir} --filelists {args.filelists}"
        cmds.append(tts_cmd)

    for log in preprocesss_pipeline(cmds):
        print(log)
        
          
    
    