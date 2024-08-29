from kfp import dsl
from kfp.dsl import Model, Dataset, Input, Output
from kfp import compiler


max_memory_limit = "14128Mi"
request_memory = "10000Mi"
cpu_limit = "22"
gpu_limit = "1"

@dsl.component
def _change_file_paths(filelist:Input[Dataset], tmp_train_dir: Input[Dataset], new_filelist: Output[Dataset]):
  
  def load_filepaths_and_text(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            yield line 
            
  with open(new_filelist.path, "w", encoding="utf-8") as f:
        f.writelines([f'{tmp_train_dir.path}/'+"|".join(line.split("|")[:2]) for line in load_filepaths_and_text(filelist.path)])

@dsl.component
def _unpack_archive_file(file : Input[Dataset], output: Output[Dataset]):
    import os, tarfile, zipfile, logging
    
    compression_type = os.path.basename(file.path).split(".")[-1]
    target = output.path     
    
    try:
        logging.info("Unpacking: %s", file.path)
        if compression_type == "tar":
            archive = tarfile.open(file.path, 'r', encoding='utf-8')
        elif compression_type == "zip":
            archive = zipfile.ZipFile(file.path, 'r')
        else:
            raise tarfile.TarError
    
        if not os.path.exists(target):
          os.makedirs(target, exist_ok=True)
        
        archive.extractall(target)
        archive.close()
    
    except (tarfile.TarError, zipfile.BadZipfile):
        raise RuntimeError("Failed to unpack archive file. The file format is not valid.")   

@dsl.container_component
def model_train(text_index:int, filelists:Input[Dataset], config:Input[Dataset], train_dir:Input[Dataset], text_cleaner:str, opt_dir:Output[Model]):
    
    return dsl.ContainerSpec(image="hankug/tts",command=["python","tts_cli.py"]
                             ,args=["--is_train","true",
                                    "--text_index",text_index,
                                    "--filelists",filelists.path,
                                    "--config",config.path
                                    ,"--text_cleaners",text_cleaner
                                    ,"--opt_dir",opt_dir.path])


@dsl.pipeline(name="tts-model-train",
              description="train tts model")
def tts_model_train(text_index:int, filelists:str, config:str, train_file:str, text_cleaner:str) -> Model:
    
    _config = dsl.importer(artifact_uri=config, artifact_class=Dataset, reimport=False).output
    _train_dir = _unpack_archive_file(file=dsl.importer(artifact_uri=train_file, artifact_class=Dataset, reimport=False).output).output
    _filelists = _change_file_paths(filelist=dsl.importer(artifact_uri=filelists, artifact_class=Dataset, reimport=False).output, tmp_train_dir=_train_dir).output

    tmp_model_path = model_train(text_index=text_index, filelists=_filelists, config=_config, train_dir=_train_dir, text_cleaner=text_cleaner)\
                            .set_accelerator_type('nvidia.com/gpu')\
                            .set_gpu_limit(gpu_limit)\
                            .set_cpu_limit(cpu_limit)\
                            .set_memory_limit(max_memory_limit)\
                            .set_memory_request(request_memory).output
                            
    return tmp_model_path
    
if __name__ == "__main__":
    compiler.Compiler().compile(tts_model_train,"./pipeline_yaml/tts_model_train_pipeline.yaml")