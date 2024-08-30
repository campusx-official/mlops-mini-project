#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 992382647488.dkr.ecr.ap-southeast-2.amazonaws.com