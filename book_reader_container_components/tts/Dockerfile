FROM python:3.8.0
WORKDIR /TTS
RUN apt update -y && apt install -y cmake && apt install -y espeak
COPY MB-iSTFT-VITS-multilingual ./

RUN echo "image build start"

RUN pip install --upgrade pip
RUN python3.8 -m pip install --upgrade numpy
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN cd /TTS && python3.8 -m pip install --no-build-isolation --ignore-installed -r requirements.txt

RUN cd /TTS/monotonic_align && mkdir monotonic_align && python setup.py build_ext --inplace && pip install --upgrade numpy && pip install protobuf==3.20.* && pip3 install numpy==1.23.5

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN sed -i '665i \ \ \ \ \ \ \ \ return torch.view_as_real(_VF.stft(input, n_fft, hop_length, win_length, window, normalized, onesided, return_complex=True))' /usr/local/lib/python3.8/site-packages/torch/functional.py && sed -i '665i \ \ \ \ if not return_complex:' /usr/local/lib/python3.8/site-packages/torch/functional.py

RUN echo "image build done"

