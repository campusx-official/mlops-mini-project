#!/bin/bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

sudo apt-get install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
sudo usermod -aG docker ubuntu
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 992382647488.dkr.ecr.ap-southeast-2.amazonaws.com