import argparse

import os
import sys
from dotenv import load_dotenv
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto
from configs.config import Config
import torch
import fairseq
import warnings
import logging

import soundfile as sf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str, required=True)
    parser.add_argument("--sid",type=str, required=True)
    parser.add_argument("--spk_item",default=0,type=int)
    parser.add_argument("--input_audio",default="",type=str)
    parser.add_argument("--vc_transform",default=6,type=int)
    parser.add_argument("--f0method",default="rmvpe",choices=["pm", "harvest", "crepe", "rmvpe"])
    parser.add_argument("--index_rate",default=0.5,type=float)
    parser.add_argument("--filter_radius",default=0.5,type=float)
    parser.add_argument("--resample_sr",default=40000,type=int)
    parser.add_argument("--rms_mix_rate",default=0.75,type=float)
    parser.add_argument("--protect",default=0.33,type=float)
    parser.add_argument("--dir_input",default="",type=str)
    parser.add_argument("--opt_input",default="",type=str)
    parser.add_argument("--batch_mode",default=True,choices=[True,False])
    
  
    args = parser.parse_args()

    now_dir = os.getcwd()
    
    sys.path.append(os.getcwd())
    
    load_dotenv()
    
    if args.model_path :
        os.environ["weight_root"] = args.model_path
        os.environ["index_root"] = f'{args.model_path}/logs'

    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    warnings.filterwarnings("ignore")
    torch.manual_seed(114514)

    def get_config():
        sys.argv = [sys.argv[0]]
        config = Config()
        print(config.python_cmd)
        return config

    config = get_config()
    vc = VC(config)

    if config.dml == True:

        def forward_dml(ctx, x, scale):
            ctx.scale = scale
            res = x.clone().detach()
            return res

        fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
        
    i18n = I18nAuto()
    logger.info(i18n)
    # 判断是否有能用来训练和加速推理的N卡
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    mem = []
    if_gpu_ok = False

    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            if any(
                value in gpu_name.upper()
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
                # A10#A100#V100#A40#P40#M40#K80#A4500
                if_gpu_ok = True  # 至少有一张能用的N卡
                gpu_infos.append("%s\t%s" % (i, gpu_name))
                mem.append(
                    int(
                        torch.cuda.get_device_properties(i).total_memory
                        / 1024
                        / 1024
                        / 1024
                        + 0.4
                    )
                )
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        default_batch_size = min(mem) // 2
    else:
        gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
        default_batch_size = 1
    gpus = "-".join([i[0] for i in gpu_infos])


    weight_root = os.getenv("weight_root")
    weight_uvr5_root = os.getenv("weight_uvr5_root")
    index_root = os.getenv("index_root")
    outside_index_root = os.getenv("outside_index_root")

    F0GPUVisible = config.dml == False

    def inference_single(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0):
        """_summary_

        Args:
            spk_item (_type_): 0~2333
            input_audio0 (_type_): 오디어 파일 경로 
            vc_transform0 (_type_): 음역대 조정 
            f0_file (_type_): _description_
            f0method0 (_type_): 음향 특징 추출 알고리즘 (["pm", "harvest", "crepe", "rmvpe"] / ["pm", "harvest", "rmvpe"])
            file_index1 (_type_): indexer 파일 경로 
            file_index2 (_type_): indexer 파일 경로 
            index_rate1 (_type_): 악센트 세기를 조절 너무 높으면 인공적인 목소리로 들림 (0~1 / 0.75)
            filter_radius0 (_type_): 평균 필터 적용 비율 잡음을 줄일 수 있음 (0~7 / 3)
            resample_sr0 (_type_): sampeling rate (0 ~ 48000)
            rms_mix_rate0 (_type_): 원복 목소리를 얼마나 반영 할지 비율 0 에 가까울 수록 원복과 비슷해짐 (0~1 / 0.25)
            protect0 (_type_): 목소리 이외의 요소 숨소리 배경음 을 얼마나 보존 할지 여부 (0~0.5 / 0.33)

        Returns:
            _type_: _description_
        """
        return vc.vc_single(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0)

    def change_vocie(sid0, protect0, protect1):
        """_summary_

        Args:
            sid0 (_type_): inferencing voice weight
            protect0 (_type_): 목소리 이외의 요소 숨소리 배경음 을 얼마나 보존 할지 여부 (0~0.5 / 0.33)
            protect1 (_type_): 목소리 이외의 요소 숨소리 배경음 을 얼마나 보존 할지 여부 (0~0.5 / 0.33)

        Returns:
            _type_: _description_
        """
        return vc.get_vc(sid0,protect0,protect1)

    def inference_multy(spk_item, dir_input, opt_input, inputs, vc_transform1, f0method1, file_index3, file_index4, index_rate2, filter_radius1, resample_sr1, rms_mix_rate1, protect1, format1):
        """_summary_

        Args:
            spk_item (_type_): 0~2333로
            dir_input (_type_): input directory 경로
            opt_input (_type_): _description_
            inputs (_type_): input file 경로
            vc_transform1 (_type_): 변환 결과물 저장 경로
            f0method1 (_type_): 음향 특징 추출 알고리즘 (["pm", "harvest", "crepe", "rmvpe"] / ["pm", "harvest", "rmvpe"])
            file_index3 (_type_): indexer 파일 경로 
            file_index4 (_type_): indexer 파일 경로 
            index_rate2 (_type_): 악센트 세기를 조절 너무 높으면 인공적인 목소리로 들림 (0~1 / 0.75)
            filter_radius1 (_type_): 평균 필터 적용 비율 잡음을 줄일 수 있음 (0~7 / 3)
            resample_sr1 (_type_): sampeling rate (0 ~ 48000)
            rms_mix_rate1 (_type_): 원복 목소리를 얼마나 반영 할지 비율 0 에 가까울 수록 원복과 비슷해짐 (0~1 / 0.25)
            protect1 (_type_): 목소리 이외의 요소 숨소리 배경음 을 얼마나 보존 할지 여부 (0~0.5 / 0.33)
            format1 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return vc.vc_multi(spk_item, dir_input, opt_input, inputs, vc_transform1, f0method1, file_index3, file_index4, index_rate2, filter_radius1, resample_sr1, rms_mix_rate1, protect1, format1)
    
    
    
    sid0 = args.sid
    _, protect0, protect1, file_index1, file_index2 = change_vocie(sid0, args.protect, args.protect)
    
    spk_item = args.spk_item
    input_audio0 = args.input_audio
    vc_transform0 = args.vc_transform
    f0_file = None
    f0method0 = args.f0method
    index_rate1 = args.index_rate
    filter_radius0 = args.filter_radius
    resample_sr0 = args.resample_sr
    rms_mix_rate0 = args.rms_mix_rate
    dir_input = args.dir_input
    opt_input = args.opt_input
    inputs =None
    
    if args.batch_mode:
        
        if dir_input != "" and opt_input != "":
        
            for log in inference_multy(spk_item, dir_input, opt_input, inputs, vc_transform0, f0method0, file_index1['value'], file_index2['value']
                        , index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect1['value'], "wav"):
                print(log)
        else :
            print("input path error")
        
    else:
        if input_audio0 != "":
            os.makedirs(args.opt_input,exist_ok=True)
            result_log, pack= inference_single(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1['value']
                                        , file_index2['value'], index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0['value'])
            print(result_log)
            name = os.path.basename(input_audio0).split(".")[0]
            sf.write(f'{args.opt_input}/{name}.wav',pack[1],pack[0],)
        else:
            print("input path error")
            
    
    
    
        
   