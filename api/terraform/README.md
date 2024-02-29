This Terraform configuration sets up an EC2 instance on AWS with a security group that allows inbound SSH access. On this
instance, a Docker container is run that exposes questions and answers in with a REST API. The endpoint has the following
format: `http://<public-ip>:8000/questions/<question-id>`. The response is a JSON object with the question, the answer 
options and the correct answer.

# Setup
Before applying this Terraform configuration, an RSA 4096-bit key pair named `aws` has to be generated and saved locally 
in the terraform directory:

```shell
ssh-keygen -t rsa -b 4096 -N '' -f aws.pem
```

To apply this configuration:

1. Ensure you have Terraform installed and configured with your AWS credentials.
2. Run `terraform init` to initialize the Terraform working directory and download the required providers.
3. Run `terraform plan` to review the changes that will be made to your infrastructure.
4. Execute `terraform apply -auto-approve` to provision the resources on AWS as defined in this configuration.
5. You will need to provide your AWS VPC and Subnet ID.


# Cleanup
To destroy the infrastructure created by this Terraform configuration, run `terraform destroy -auto-approve`. This will 
remove all resources created by this configuration from your AWS account. You will need to provide your AWS VPC and 
Subnet ID. Afterward, you can delete the key pair.


