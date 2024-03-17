FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt update -y && apt install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY finetune.py .

CMD accelerate launch finetune.py