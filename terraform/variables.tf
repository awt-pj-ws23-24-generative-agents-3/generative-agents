variable "path_to_key" {
  type        = string
  description = "Path to the private key associated with the public key attached to the Prometheus instance"
  default = "aws.pem"
}

variable "public_key" {
  type        = string
  description = "Public key to be attached to the instance"
  default = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDLSGxxRM1DPSa5RZWDdNAsWrvXHhbz7qeQau3ANdv6Z7IFNWxkFVRlfOqY4PJHa84+eVuoIGT4FpAmRcSlGWw42kRxlMftOzQh3o1zKT89rlgtjtwOct2xXIjG+0fAFwkPQ82mbLGciuawEcQFQcODSMm8TtVWoouZ3CzoCBR6/sqKLQHsQ7fDvNTMnYSIk2iLqngw1uNyCAT0jnN1/YaaJM7m4axz2OI5x7Y3F+s9+vZZ2qGwE46zHJA/MidZmZywC97Z2S5ccnZBR/FUNAo3Oy625QNTJseMvn/9/IY56NwWqJtDvz7WdTT1AYgVJjVYoR281lllSxfL5zLqD9RVz4jhqgFNB4lz3FYrgRzjite7cdlZXGzdOWRdSp5Y/AgnBJ8z6uiogA9eQ0sAS9bVZur5br0wT72Uh8389BChiwshkzMUfUNIGtejqb8hJF1l7M7AbWKvcaeqWTLSg5etFdGB1D375o/SvQbh/8CR4r6I4vZZx2oosMmW8cSZjyt/HuACq+XgHEmdoQ3szyqdqGqrAHOJkxsRbdE9XOI0dkNzYOaJR7AHizhzcWlo/iLwDvGc8G+/2ilQ4y+zRu5+DsemvCXgnSpiWNI/3MTM8vKubhILU3AGSG3hxBLVkbalCOjV9fzOfJIOMOrWUbrkcAZt793WZG161FQea5pFgQ== Siar@MacBook-Pro-929.fritz.box"
}

