variable "vpc_id" {
  type        = string
  description = "AWS VPC ID (using default is recommended)"
  default = "vpc-0f2d9adf03961f221"
}

variable "subnet_id" {
  type        = string
  description = "AWS Subnet ID (using default is recommended)"
    default = "subnet-028b97041fb0039d2"
}

