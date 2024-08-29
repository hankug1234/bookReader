import soundfile as sf
import bson


def load_audio_to_bytes(audio_path):
        audio, samplerate = sf.read(audio_path)
        return audio.tolist()

def make_sample_input():
  datas = {}
  datas["scripts"] = ["그것은 진정한 기사의 임무이자 의무. 아니! 의무가 아니라, 특권이노라.",
                      "불가능한 꿈을 꾸는 것.",
                      "무적의 적수를 이기며,",
                      "견딜 수 없는 고통을 견디고,",
                      "고귀한 이상을 위해 죽는 것."]
    
  with open("./data.json", "wb") as outfile:
    outfile.write(bson.dumps(datas))        


if __name__ == "__main__":
  make_sample_input()