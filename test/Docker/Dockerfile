FROM ubuntu:20.04
COPY mnist /exp/mnist
COPY requirements.txt /exp/requirements.txt
RUN apt-get clean
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp/mnist/api
CMD ["flask","run","--host=0.0.0.0","-p","8000"]
