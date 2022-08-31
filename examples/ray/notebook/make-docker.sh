#!/bin/sh
docker build -t coreystaten/raynotebook:latest --build-arg NGROK_AUTH_TOKEN=$NGROK_AUTH_TOKEN .
