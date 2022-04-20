FROM python:3.9

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev wget
RUN pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /code
ADD . .
CMD /bin/bash entrypoint.sh
