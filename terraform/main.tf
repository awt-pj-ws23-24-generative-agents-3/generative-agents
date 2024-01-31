provider "aws" {
  region = "us-east-2"
}

terraform {
  required_providers {
    aws = {
      version = "~>5.0"
      source  = "hashicorp/aws"
    }
  }
}

resource "aws_key_pair" "deployer" {
  key_name   = "aws"
  public_key = var.public_key
}

# Create a security group
resource "aws_security_group" "my-security-group-csb" {
  name   = "my-security-group"
  vpc_id = "vpc-0ecaaa86c9a76e267"

  # Allow inbound SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow inbound HTTP
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "questions_api" {
  ami           = "ami-0c758b376a9cf7862" # Debian 12 64-bit (Arm), username: admin
  instance_type = "t4g.nano"
  subnet_id     = "subnet-034cd218e2b28c58a"
  vpc_security_group_ids = [aws_security_group.my-security-group-csb.id]
  key_name = aws_key_pair.deployer.key_name

  user_data = <<-EOT
    #!/bin/bash

    sudo apt update
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo mkdir -p /app  # Create a directory on the instance (if not already present)
    sudo chown -R admin:admin /app  # Change ownership to the desired user (replace 'admin' with the actual username)

    # Pull the Docker image from the registry
    sudo docker pull ghcr.io/siar-akbayin/questions-api:latest


    # Build and run Docker container
    sudo docker run -d -p 8000:8000 --name questions-api ghcr.io/siar-akbayin/questions-api:latest
    EOT
  tags = {
    Name = "Questions API"
  }
}
