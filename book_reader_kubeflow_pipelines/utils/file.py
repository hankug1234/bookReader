import shutil
import os


def get_filename(path):
    return os.path.basename(path)


def move_file(src,dst,file_name):
    shutil.copyfile(src, os.path.join(dst, file_name))


def make_dir(dir_path):
    os.makedirs(dir_path,exist_ok=True)


