# An Application of Large Language Models and Generative Agents to Compare Quality and Quantity Learning Processes and Performance

This project is done for the module "Project Advanced Web Technologies". It contains all our deliverables including the 
presentation slides and videos in the presentations folder, the final report and the figures in the report folder, the 
results in the results folder and all the other folders are the code, the data and the terraform files.

The purpose of this project is to explore the innovative application of generative agents powered by Large Language Models (LLMs), specifically
focusing on Llama2, to facilitate learning in American History. By leveraging advanced LLM capabilities, we simulate two 
distinct learning methodologies—quantitative and qualitative—for agents and assess their effectiveness through standardized 
AP U.S. History test questions.

# Setup

There are three different ways to set up this project:

## Local Setup without Docker

To set up the project locally, you need to have Python 3.8 or higher installed. Before running the agents-interaction script,
the API need to be set up before. To do so, you need to go to the api folder and run the following commands:

```shell
pip install -r requirements.txt
python app.py
```

Now the endpoint will be available on localhost:8000/questions/<question-id>. Afterward, you can go to the agents-interaction 
and run the following commands:

```shell
pip install -r requirements.txt
python main.py
```

## Local Setup with Docker

To set up the project locally with Docker, you need to have Docker installed. Before running the agents-interaction script,
the API need to be set up before. To do so, you need to go to the api folder and run the following commands:

```shell
docker build -t <your-image-name>.
docker run -p 8000:8000 --name <your-image-name> 
```
Afterward, you can go to the agents-interaction and run the following commands:

```shell
docker build -t <your-image-name> .
docker run -p 8088:8088 --name <your-image-name> 
```

To get the generated files, you need to copy them from the container to your local machine. To do so, you can run the following
commands:

```shell
docker cp <CONTAINER_ID>:/usr/src/app/ ./results-local/  
```
This will get you the whole project folder, and you can extract the necessary files manually.

## AWS Setup with Terraform
First follow the instructions in the README.md file in the api/terraform folder. Afterward, you need to follow the instructions
of the README.md file in the agents-interaction/terraform folder. Do not forget to update the API URL and to use your own 
container registry in the main.tf file with the image that uses the API URL.

