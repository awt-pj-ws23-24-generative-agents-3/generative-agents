This Terraform configuration sets up an EC2 instance on AWS with a security group that allows inbound SSH access. On this
instance, a Docker container is pulled and run which runs the Python script that generates the agents, runus the practice 
phase and the exam phase and stores the results in different text files. They are then copied from the container to the instance
and afterward the Terraform script stores them locally. 

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


