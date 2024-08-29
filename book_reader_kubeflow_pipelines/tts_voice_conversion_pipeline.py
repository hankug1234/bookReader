from kfp import dsl
from kfp.dsl import Dataset, Model, Output, Input
from kfp import compiler

max_memory_limit = "14128Mi"
request_memory = "5000Mi"
cpu_limit = "8"
gpu_limit = "1"


@dsl.container_component
def text_to_sound(is_train:str, text_index:int, filelists:Input[Dataset], config:Input[Dataset], checkpoint: Input[Model],
                  text_cleaner:str, opt_dir:Output[Dataset]):
    
    return dsl.ContainerSpec(image="hankug/tts",command=["python","tts_cli.py"]
                             ,args=["--is_train",is_train,
                                    "--text_index",text_index,
                                    "--filelists",filelists.path,
                                    "--config",config.path
                                    ,"--text_cleaners",text_cleaner
                                    ,"--opt_dir",opt_dir.path
                                    ,"--checkpoint", checkpoint.path])


@dsl.container_component
def sound_conversion(sid:str, spk_item:int, vc_transform:int, f0method:str, index_rate:float, model_path: Input[Model],
                     filter_radius:float, resample_sr:int, rms_mix_rate:float, protect:float, dir_input: Input[Dataset], opt_input:Output[Dataset]):
    
    return dsl.ContainerSpec(image="hankug/vc",command=["python","vc_conversion.py"]
                             ,args=["--model_path", model_path.path, 
                                    "--sid", sid,
                                    "--spk_item",  spk_item,
                                    "--vc_transform", vc_transform,
                                    "--f0method", f0method,
                                    "--index_rate", index_rate,
                                    "--filter_radius", filter_radius,
                                    "--resample_sr", resample_sr,
                                    "--rms_mix_rate", rms_mix_rate,
                                    "--protect", protect,
                                    "--dir_input", dir_input.path,
                                    "--opt_input", opt_input.path])
    
@dsl.pipeline(name="tts-voice-conversion",
              description="converse tts ouput voice to other speaker's voice")
def tts_voice_conversion(text_index:int, scripts_path:str, config_path:str,
                         text_cleaner:str, tts_model_path:str, rvc_model_path:str, 
                         sid:str, spk_item:int, vc_transform:int, f0method:str, index_rate:float,
                         filter_radius:float, resample_sr:int, rms_mix_rate:float, protect:float):
    
    
    tts_model = dsl.importer(artifact_uri=tts_model_path, artifact_class=Model, reimport=False).output
    config = dsl.importer(artifact_uri=config_path, artifact_class=Dataset, reimport=False).output
    scripts = dsl.importer(artifact_uri=scripts_path, artifact_class=Dataset, reimport=False).output

    audios = text_to_sound(is_train="false", text_index=text_index, filelists=scripts
                           ,config=config,checkpoint=tts_model,text_cleaner = text_cleaner)\
                            .set_accelerator_type('nvidia.com/gpu')\
                            .set_gpu_limit(gpu_limit)\
                            .set_cpu_limit(cpu_limit)\
                            .set_memory_limit(max_memory_limit)\
                            .set_memory_request(request_memory)
    
    rvc_model = dsl.importer(artifact_uri=rvc_model_path, artifact_class=Model, reimport=False).output
    
    sound_conversion(sid=sid, spk_item=spk_item, vc_transform=vc_transform, f0method=f0method,
                     index_rate=index_rate, model_path=rvc_model,
                     filter_radius=filter_radius, resample_sr=resample_sr,
                     rms_mix_rate=rms_mix_rate, protect=protect, dir_input=audios.outputs["opt_dir"])\
                     .set_accelerator_type('nvidia.com/gpu')\
                     .set_gpu_limit(gpu_limit)\
                     .set_cpu_limit(cpu_limit)\
                     .set_memory_limit(max_memory_limit)\
                     .set_memory_request(request_memory)
    
    

if __name__ == "__main__":
    compiler.Compiler().compile(tts_voice_conversion,"./pipeline_yaml/tts_voice_conversion_pipeline.yaml")