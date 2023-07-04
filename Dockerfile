FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

WORKDIR /app

COPY requirements_server.txt .
RUN pip3 install -r requirements_server.txt

COPY ./server.py .
COPY ./model.pth .

EXPOSE 7777

CMD ["python3", "server.py"]