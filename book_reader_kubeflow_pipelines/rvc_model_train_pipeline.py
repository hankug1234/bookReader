from kfp import dsl
from kfp.dsl import Model, Dataset, Input, Output
from kfp import compiler


max_memory_limit = "14128Mi"
request_memory = "10000Mi"
cpu_limit = "22"
gpu_limit = "1"


@dsl.container_component
def model_train(export_path:Output[Model], exp_dir:str, sr:str, train_dir:Input[Dataset], np:int, save_epoch:int, total_epoch:int,
                     batch_size:int, gpus:str, gpus_rmvpe:str):
    
    return dsl.ContainerSpec(image="hankug/vc",command=["python","vc_train.py"]
                             ,args=["--export_path", export_path.path,
                                    "--exp_dir", exp_dir,
                                    "--train_dir", train_dir.path,
                                    "--np", np,
                                    "--save_epoch", save_epoch,
                                    "--total_epoch", total_epoch,
                                    "--f0method", "rmvpe_gpu",
                                    "--batch_size", batch_size,
                                    "--gpus", gpus,
                                    "--gpus_rmvpe", gpus_rmvpe])


@dsl.pipeline(name="rvc-model-train", description="train rvc model")
def rvc_model_train(exp_dir:str, sr:str, train_dir_path:str, np:int, save_epoch:int, total_epoch:int,
                     batch_size:int, gpus:str, gpus_rmvpe:str) -> Model:
    
    train_dir = dsl.importer(artifact_uri=train_dir_path, artifact_class=Dataset, reimport=False).output

    tmp_model_path = model_train(exp_dir=exp_dir, sr=sr, train_dir=train_dir, np=np, save_epoch=save_epoch, total_epoch=total_epoch,
                     batch_size=batch_size, gpus=gpus, gpus_rmvpe=gpus_rmvpe)\
                            .set_accelerator_type('nvidia.com/gpu')\
                            .set_gpu_limit(gpu_limit)\
                            .set_cpu_limit(cpu_limit)\
                            .set_memory_limit(max_memory_limit)\
                            .set_memory_request(request_memory).output
    return tmp_model_path

if __name__ == "__main__":
    compiler.Compiler().compile(rvc_model_train,"./pipeline_yaml/rvc_model_train_pipeline.yaml")
    