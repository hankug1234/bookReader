from pydub import AudioSegment
import json

def load_audio(audio_path):
    return AudioSegment.from_file(audio_path)

def audio_clip(audio,start,end):
    trimmed_audio = audio[start*1000:end*1000]
    return trimmed_audio


def save_audio_clip(clip,save_path,format):
    clip.export(save_path,format=format)


if __name__ == "__main__":
    script_path = "../wisper-script/script_result.json"
    audio_path = "../vocal-remover/fate-zero-1_Vocals.mp3"
    audio = load_audio(audio_path)

    with open(script_path,'r',encoding="UTF-8") as fp:
        json = json.load(fp)
        segments = json["segments"]
        for segment in segments:
            print(f"{segment['start']} -> {segment['end']} : {segment['text']}")
            clip = audio_clip(audio,segment['start'],segment['end'])
            if segment['no_speech_prob'] < 0.5:
                save_audio_clip(clip,f"./less_criteria/{segment['text']}.mp3","mp3")
            else:
                save_audio_clip(clip,f"./more_criteria/{segment['text']}.mp3","mp3")



