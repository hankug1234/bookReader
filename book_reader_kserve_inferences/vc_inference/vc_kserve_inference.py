from kserve import Model, ModelServer, InferResponse, InferOutput
from typing import Dict
from dotenv import load_dotenv
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto
from configs.config import Config
import torch,fairseq,warnings,logging,os,sys,json, argparse, bson
import soundfile as sf
import numpy as np

def bson_parsing(data):
    data = bson.loads(data)
    data["audios"] = [np.array(audio) for audio in data['audios']]
    return data


class RvcModel(Model):
    def __init__(self,name:str):
        super().__init__(name)
        self.name = name
        self.model_path = os.environ.get("MODEL_PATH", "")
        self.sid = f'{os.path.basename(os.environ.get("MODEL_PATH", "")).split(".")[0]}.pth'
        self.load()
    
    def load(self):
        
        self.ready = True
        print(f'model_path : {os.environ.get("MODEL_PATH", "")}')
        self.now_dir = os.getcwd()
        
        sys.path.append(os.getcwd())
        
        load_dotenv()
        
        if self.model_path :
            os.environ["weight_root"] = self.model_path
            os.environ["index_root"] = f'{self.model_path}/logs'

        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.logger = logging.getLogger(__name__)

        warnings.filterwarnings("ignore")
        torch.manual_seed(114514)

        def get_config():
            sys.argv = [sys.argv[0]]
            config = Config()
            print(config.python_cmd)
            return config

        self.config = get_config()
        self.vc = VC(self.config)

        if self.config.dml == True:

            def forward_dml(ctx, x, scale):
                ctx.scale = scale
                res = x.clone().detach()
                return res

            fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
            
        i18n = I18nAuto()
        self.logger.info(i18n)
        self.ngpu = torch.cuda.device_count()
        self.gpu_infos = []
        self.mem = []
        self.if_gpu_ok = False

        if torch.cuda.is_available() or self.ngpu != 0:
            for i in range(self.ngpu):
                self.gpu_name = torch.cuda.get_device_name(i)
                if any(
                    value in self.gpu_name.upper()
                    for value in [
                        "10",
                        "16",
                        "20",
                        "30",
                        "40",
                        "A2",
                        "A3",
                        "A4",
                        "P4",
                        "A50",
                        "500",
                        "A60",
                        "70",
                        "80",
                        "90",
                        "M4",
                        "T4",
                        "TITAN",
                        "4060",
                        "L",
                        "6000",
                    ]
                ):
                    self.if_gpu_ok = True 
                    self.gpu_infos.append("%s\t%s" % (i, self.gpu_name))
                    self.mem.append(
                        int(
                            torch.cuda.get_device_properties(i).total_memory
                            / 1024
                            / 1024
                            / 1024
                            + 0.4
                        )
                    )
        if self.if_gpu_ok and len(self.gpu_infos) > 0:
            self.gpu_info = "\n".join(self.gpu_infos)
            self.default_batch_size = min(self.mem) // 2
        else:
            self.gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
            self.default_batch_size = 1
        self.gpus = "-".join([i[0] for i in self.gpu_infos])


        self.weight_root = os.getenv("weight_root")
        self.weight_uvr5_root = os.getenv("weight_uvr5_root")
        self.index_root = os.getenv("index_root")
        self.outside_index_root = os.getenv("outside_index_root")

        self.F0GPUVisible = self.config.dml == False
        
        
        self.sid0 = self.sid
        _, self.protect0, self.protect1, self.file_index1, self.file_index2 = self.vc.get_vc(self.sid0, 0.33, 0.33)
        
    
    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        
        if os.path.exists(self.model_path) != True:
            return {"prediction" : "model not found", "code":404}
        
        if headers['content-type'] == 'application/x-www-form-urlencoded':
            datas = bson_parsing(payload)
        else:
            return {"prediction" : "must content-type = application/x-www-form-urlencoded", "code":500}
        
        spk_item = datas["spk_item"]
        audios = datas["audios"]
        vc_transform0 = datas["vc_transform"]
        f0_file = None
        f0method0 = datas["f0method"]
        index_rate1 = datas["index_rate"]
        filter_radius0 = datas["filter_radius"]
        resample_sr0 = datas["resample_sr"]
        rms_mix_rate0 = datas["rms_mix_rate"]
        
        if type(spk_item) != int : 
              return {"prediction" : "spk_item must int", "code":404}
        elif len(audios) <= 0 :
             return {"prediction" : " audios must more then 0", "code":404}
        elif type(vc_transform0) != int : 
             return {"prediction" : "vc_transform must int", "code":404}
        elif f0method0 not in ["pm", "harvest", "crepe", "rmvpe"]:
            return {"prediction" : "f0method must in pm, harvest, crepe, rmvpe", "code":404}
        elif type(index_rate1) != float: 
            return {"prediction" : "index_rate must float", "code":404}
        elif type(filter_radius0) != int: 
            return {"prediction" : "filter_radius must int" , "code":404}
        elif type(resample_sr0) != int: 
            return {"prediction" : "resample_sr must int" , "code":404}
        elif type(rms_mix_rate0) != float: 
            return {"prediction" : "rms_mix_rate must float" , "code":404}
        
        
        os.makedirs("/temp_audios",exist_ok=True)
        audio_paths = []
        for i,audio in enumerate(audios):
            sf.write(f'/temp_audios/audio{i}.wav',audio,resample_sr0,)
            audio_paths.append(f'/temp_audios/audio{i}.wav')
        
        conversion_results = []
        try:
            for audio_path in audio_paths:
                result_log, pack= self.vc.vc_single(spk_item, audio_path, vc_transform0, f0_file, f0method0, self.file_index1['value']
                                            , self.file_index2['value'], index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, self.protect0['value'])
        
                conversion_results.append({"sr":pack[0],"audio":pack[1].tolist(),"dtype":str(pack[1].dtype)})
                
            return bson.dumps({"prediction" : conversion_results ,"log":result_log, "code":200})
        except Exception as e:
            return {"prediction" : f"{e}", "code":404}

if __name__ == '__main__':
    model = RvcModel("rvc")
    ModelServer().start([model])