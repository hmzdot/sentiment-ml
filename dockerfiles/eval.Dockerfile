FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ADD . /app

RUN pip install --upgrade pip && pip install .

CMD ["python", "src/eval.py"]