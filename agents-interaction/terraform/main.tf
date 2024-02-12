provider "aws" {
  region = "us-east-2"
}

terraform {
  required_providers {
    aws = {
      version = "~> 5.0"
      source  = "hashicorp/aws"
    }
  }
}

resource "aws_key_pair" "deployer" {
  key_name   = "aws"
  public_key = file("${path.module}/aws.pem.pub")
}

resource "aws_security_group" "my_security_group_awt_agents" {
  name   = "my-security-group-agents"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8088
    to_port     = 8088
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

resource "aws_instance" "generative_agents" {
  ami                      = "ami-0e6e78596f3522ace"
  instance_type            = "p3.2xlarge"
  subnet_id                = var.subnet_id
  vpc_security_group_ids   = [aws_security_group.my_security_group_awt_agents.id]
  key_name                 = aws_key_pair.deployer.key_name

  tags = {
    Name = "Generative Agents"
  }
}

resource "aws_ebs_volume" "extra_storage" {
  availability_zone = aws_instance.generative_agents.availability_zone
  size              = 50
  type              = "gp3"
  tags = {
    Name = "Extra Storage"
  }
}

resource "aws_volume_attachment" "extra_storage_attachment" {
  device_name = "/dev/sdh"
  volume_id   = aws_ebs_volume.extra_storage.id
  instance_id = aws_instance.generative_agents.id
}

resource "terraform_data" "generative_agents_setup" {
  depends_on = [aws_volume_attachment.extra_storage_attachment]

  connection {
    type        = "ssh"
    user        = "ec2-user"
    private_key = file("${path.module}/${aws_key_pair.deployer.key_name}.pem")
    host        = aws_instance.generative_agents.public_ip
  }

  provisioner "remote-exec" {
    inline = [
      "sudo apt update",
      "sudo apt install -y apt-transport-https ca-certificates curl software-properties-common",
      "curl -fsSL https://get.docker.com -o get-docker.sh",
      "sudo sh get-docker.sh",
      "sudo usermod -aG docker $USER",

      "sudo systemctl stop docker",
      "sudo mkfs.ext4 /dev/sdh",
      "sudo mkdir -p /mnt/docker_data",
      "sudo mount /dev/sdh /mnt/docker_data",
      "echo '/dev/sdh /mnt/docker_data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab",
      "echo '{\"data-root\": \"/mnt/docker_data/docker\"}' | sudo tee /etc/docker/daemon.json",
      "sudo systemctl start docker",
      "sudo systemctl enable docker",
      "sudo mkdir -p /app",
      "sudo chown -R ec2-user:ec2-user /app",

      "sudo docker pull ghcr.io/siar-akbayin/generative-agents:latest",
      "sleep 20",
      "sudo docker run -d -p 8088:8088 --name generative-agents ghcr.io/siar-akbayin/generative-agents:latest",
      "sudo docker restart generative-agents"
    ]
  }
}
