output "api_endpoint" {
  value = "http://${aws_instance.questions_api.public_ip}:8000/questions/1"
}
