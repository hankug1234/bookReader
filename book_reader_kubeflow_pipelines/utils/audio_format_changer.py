#pip install pydub  
## install ffmpeg to your computer  
# apt-get install ffmpeg  
from pydub import AudioSegment  
from pydub.playback import play


def mp3ToWav(src_file, dest_file):  
    sound = AudioSegment.from_mp3(src_file)

    #play(sound)  
    sound.export(dest_file, format="wav")  
    pass  

def wavToMp3(src_file, dest_file):  
    sound = AudioSegment.from_wav(src_file)  
    sound.export(dest_file, format="mp3")