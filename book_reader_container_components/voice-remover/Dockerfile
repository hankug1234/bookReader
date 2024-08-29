FROM python:3.9.9

WORKDIR VOICEREMOVER
RUN apt -y update && apt -y upgrade
RUN apt -y install curl && apt -y install zip
RUN curl -L https://github.com/tsurumeso/vocal-remover/releases/download/v5.1.0/vocal-remover-v5.1.0.zip > vocal-remover-v5.1.0.zip 
RUN unzip vocal-remover-v5.1.0.zip -d ./
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN cd vocal-remover && pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
ADD ./inference.py /VOICEREMOVER/vocal-remover/
WORKDIR ./vocal-remover
