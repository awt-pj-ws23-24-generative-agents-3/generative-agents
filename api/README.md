This directory contains the code for the API which contains the questions and answers for the agents-interaction project. 
The endpoint has the following format: `http://<public-ip>:8000/questions/<question-id>`. The response is a JSON object with
the question, the answer options and the correct answer.

The JSON of question 1, for instance, looks like this:
```json
{
  "Choice_A": "(A) establish a place where they could practice their religion freely.",
  "Choice_B": "(B) find an all-water route to the East.",
  "Choice_C": "(C) end the practice of primogeniture.",
  "Choice_D": "(D) spread Christianity around the world.",
  "Choice_E": "(E) uplift and civilize other peoples of the world.",
  "Correct_Answer": "B",
  "ID": 1,
  "Question": "The discovery of the New World resulted from the desire of many Europeans to"
}
```

## Setup
To set up with Docker or locally, check the root README.md file. To set up with Terraform on AWS, check the terraform 
directory in this directory.
