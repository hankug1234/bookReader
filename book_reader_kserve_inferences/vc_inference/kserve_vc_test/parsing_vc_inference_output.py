import bson, numpy
import soundfile as sf

def bson_parsing(data):
        data = bson.loads(data)
        audios = data["prediction"]
        audios = [(numpy.array(audio["audio"],dtype=numpy.int16), audio["sr"]) for audio in audios]
        return audios

def load_bson_file(file_path):
    with open(file_path, 'rb') as f:
      audios = bson_parsing(f.read())
      for i,audio in enumerate(audios):
            sf.write(f"audio{i}.wav",audio[0],audio[1])


if __name__ == "__main__":
    load_bson_file("./output.json")