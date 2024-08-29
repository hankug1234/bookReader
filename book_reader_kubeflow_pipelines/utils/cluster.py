from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from glob import glob
from numpy import dot
from numpy.linalg import norm
from data_preprocessing_pipline.utils import file
import os


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
    file.make_dir(save_path)
    for i,cluster in enumerate(clusters):
        cluster_path = save_path+os.sep+f"cluster{i}"
        file.make_dir(cluster_path)
        for path in cluster:
            filename = file.get_filename(path)
            file.move_file(path,cluster_path,filename)



if __name__ == "__main__":
    dir = "../voice-extract/less_criteria/*"
    result = voice_clustering(dir,0.8)
    save_result(result,"./result")



