FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-ecee7c1
RUN pip install ray pyarrow
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
    tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
    tee /etc/apt/sources.list.d/ngrok.list && \
    apt update && apt install ngrok
ARG NGROK_AUTH_TOKEN
ENV NGROK_AUTH_TOKEN=$NGROK_AUTH_TOKEN
RUN ngrok config add-authtoken $NGROK_AUTH_TOKEN
COPY . /raynotebook
RUN pip install /raynotebook
