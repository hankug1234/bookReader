from kfp import dsl
from kfp.dsl import Dataset, Input, Output
from kfp import compiler

max_memory_limit = "14128Mi"
request_memory = "5000Mi"
cpu_limit = "8"
gpu_limit = "1"

@dsl.container_component
def whisper(context: Input[Dataset], vocal_paths: str, model: str, language: str, format: str, output_dir2: Output[Dataset]):
    return dsl.ContainerSpec(image="hankug/book-whisper",command=["sh","-c", "$0"]
                             ,args=[f"whisper {vocal_paths} --model {model} --language {language} --output_dir {output_dir2.path} --output_format {format}"])

@dsl.container_component
def vocal_remover(context: Input[Dataset], audio_paths: str ,output_dir1: Output[Dataset]):
    return dsl.ContainerSpec(image="hankug/voice-remover"
                             ,command=["sh","-c", "$0"]
                             ,args=[f"python3 inference.py --inputs {audio_paths} --output_dir {output_dir1.path} --gpu 0"])
    
@dsl.component
def get_audio_files(audio_dir: Input[Dataset]) -> str:
    import os
    return " ".join([f"{audio_dir.path}{os.sep}{path}" for path in os.listdir(audio_dir.path)])
    
@dsl.component
def extract_voical_file(audio_dir: Input[Dataset]) -> str:
    import os

    voice_paths = [path for path in os.listdir(audio_dir.path) if "_Vocals.wav" in path]
    if len(voice_paths) == 0:
        raise Exception
    else:
        return " ".join([f"{audio_dir.path}{os.sep}{voice_path}" for voice_path in voice_paths])

@dsl.component(packages_to_install=["pydub"])
def voice_cliper(scripts_path: Input[Dataset], audios_path: Input[Dataset],speech_prob:float, voices_clips: Output[Dataset]):
    from pydub import AudioSegment
    import json, os

    def load_audio(audio_path):
        return AudioSegment.from_file(audio_path)

    def audio_clip(audio,start,end):
        trimmed_audio = audio[start*1000:end*1000]
        return trimmed_audio


    def save_audio_clip(clip,save_path,format):
        clip.export(save_path,format=format)

    audio_paths = [f"{audios_path.path}{os.sep}{path}" for path in os.listdir(audios_path.path) if "_Vocals.wav" in path]
    
    os.makedirs(voices_clips.path,exist_ok=True)
    
    for audio_path in audio_paths:
    
        audio = load_audio(audio_path)
        basename = os.path.basename(audio_path).split(".")[0]
        script_path = f"{scripts_path.path}{os.sep}{basename}.json"

        with open(script_path,'r',encoding="UTF-8") as fp:
            script = json.load(fp)
            segments = script["segments"]
            for segment in segments:
                try:
                    print(f"{segment['start']} -> {segment['end']} : {segment['text']}")
                    clip = audio_clip(audio,segment['start'],segment['end'])
                    if segment['no_speech_prob'] < speech_prob:
                        save_audio_clip(clip,f"{voices_clips.path}/{str(segment['text']).strip()}.wav","wav")
                except Exception as e:
                    print(e)


@dsl.component
def voice_cluster(voice_clips: Input[Dataset], cluster_criteria: float, voice_clusters: Output[Dataset]):
   
    import os,shutil
    requirements = [
        "numpy", 
        "Resemblyzer"
    ]
    os.system("pip install --upgrade pip")
    for package in requirements:
        os.system(f"pip install {package}")
   
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pathlib import Path
    from glob import glob
    from numpy import dot
    from numpy.linalg import norm

    def get_filename(path):
        return os.path.basename(path)


    def move_file(src,dst,file_name):
        shutil.copyfile(src, os.path.join(dst, file_name))


    def make_dir(dir_path):
        os.makedirs(dir_path,exist_ok=True)


    def cosine_similarity(embed1,embed2):
        return dot(embed1,embed2) / (norm(embed1)*norm(embed2))


    def voice_embed(voice_path):
        fpath = Path(voice_path)
        wav = preprocess_wav(fpath)
        encoder = VoiceEncoder()
        return encoder.embed_utterance(wav)


    def voice_clustering(dir,threshold):
        paths = glob(dir)
        embeds = [voice_embed(path) for path in paths]
        metrix = [[cosine_similarity(embed1, embed2) for embed2 in embeds] for embed1 in embeds]

        cluster_list = []
        clustered_index = set()
        result = []

        for i,row in enumerate(metrix):
            cluster_set = [i for i,value in enumerate(row) if value > threshold]
            cluster_size = sum(cluster_set)
            cluster_list.append((cluster_set,cluster_size,i))

        sorted_cluster_list = sorted(cluster_list,key=lambda x: -x[1])

        for cluster in sorted_cluster_list:
            if cluster[2] not in clustered_index:
                new_set = set(cluster[0])
                diff = new_set.difference(clustered_index)
                origine = len(cluster[0])
                if len(diff) > origine/2:
                    clustered_index.update(cluster[0])
                    result.append([paths[index] for index in cluster[0]])

        return result


    def save_result(clusters,save_path):
        make_dir(save_path)
        
        for i,cluster in enumerate(clusters):
            cluster_path = save_path+os.sep+f"cluster{i}"
            voice_list = []
            make_dir(cluster_path)
            for path in cluster:
                filename = get_filename(path)
                move_file(path,cluster_path,filename)
                voice_list.append(f"{cluster_path}{os.sep}{filename}|{filename.split('.')[0]}")
                
            with open(f"{cluster_path}{os.sep}voice{i}.txt","w",encoding="utf-8") as f:
                f.write("\n".join(voice_list))

    result = voice_clustering(f"{voice_clips.path}/*",cluster_criteria)
    save_result(result,voice_clusters.path)


@dsl.component
def return_save_path(save_path: Input[Dataset]) -> str:
    return save_path.path
    

@dsl.pipeline(name="voice-extractor",
              description="extract voice from input audio file")
def voice_extract(audio_dir_path: str, model: str, language: str, speech_prob:float, cluster_criteria:float) -> str:

    audio_dir = dsl.importer(artifact_uri=audio_dir_path, artifact_class=Dataset, reimport=False).output
    audio_paths = get_audio_files(audio_dir=audio_dir).output    
    
    remover = vocal_remover(context=audio_dir ,audio_paths=audio_paths).set_accelerator_type('nvidia.com/gpu')\
    .set_gpu_limit(gpu_limit)\
    .set_cpu_limit(cpu_limit)\
    .set_memory_limit(max_memory_limit)\
    .set_memory_request(request_memory).output
    
    pure_vocal_file_paths = extract_voical_file(audio_dir=remover).output
    
    scripts = whisper(context=remover, vocal_paths=pure_vocal_file_paths, model=model,language=language,format="json")\
    .set_accelerator_type('nvidia.com/gpu')\
    .set_gpu_limit(gpu_limit)\
    .set_cpu_limit(cpu_limit)\
    .set_memory_limit(max_memory_limit).output
    
    voice_clips = voice_cliper(scripts_path=scripts, audios_path=remover, speech_prob=speech_prob).output
    
    voice_clusters = voice_cluster(voice_clips=voice_clips, cluster_criteria=cluster_criteria).output
    
    return return_save_path(save_path = voice_clusters).output


if __name__ == "__main__":
    compiler.Compiler().compile(voice_extract,"./pipeline_yaml/voice_extract_pipeline.yaml")