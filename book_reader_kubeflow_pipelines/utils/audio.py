import os
from moviepy.editor import VideoFileClip


def convert_video_to_audio_moviepy(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")


def extract_video_clip(video_file):
    return VideoFileClip(video_file)


def save_video_clip(clip, path, file_name, output_ext="mp3"):
    clip.audio.write_audiofile(f"{path}{os.sep}{file_name}.{output_ext}")


if __name__ == "__main__":
    vf = "/home/hankug/about_tts/videos/Fate／Zero 01화.mp4"
    convert_video_to_audio_moviepy(vf)