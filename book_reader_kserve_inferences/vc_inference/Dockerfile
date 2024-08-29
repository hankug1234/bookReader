FROM hankug/vc:latest
RUN pip install kserve && pip install bson && pip install pydantic==2.2
COPY ./vc_kserve_inference.py /app/
CMD ["python", "vc_kserve_inference.py"]
