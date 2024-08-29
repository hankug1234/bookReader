FROM hankug/tts:latest
RUN pip install kserve && pip install bson && pip install pydantic==2.2
COPY ./tts_kserve_inference.py /TTS/
CMD ["python", "tts_kserve_inference.py"]