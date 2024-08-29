import soundfile as sf
import bson


def load_audio_to_bytes(audio_path):
        audio, samplerate = sf.read(audio_path)
        return audio.tolist()

def make_sample_input():
  datas = {}
  datas["spk_item"] = 0
  datas["audios"] = [load_audio_to_bytes("/home/hankug/음악/source.wav")]
  datas["vc_transform"] = 6
  datas["f0method"] = "rmvpe"
  datas["index_rate"] = 0.75
  datas["filter_radius"] = 3
  datas["resample_sr"] = 40000
  datas["rms_mix_rate"] = 0.25
    
  with open("./data.json", "wb") as outfile:
    outfile.write(bson.dumps(datas))        


if __name__ == "__main__":
  make_sample_input()
 