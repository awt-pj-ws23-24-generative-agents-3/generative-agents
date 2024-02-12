import requests
import random


def fetch_question_from_api(question_id):
    """Fetch a question from the API."""
    api_url = f'http://3.139.84.244:8000/questions/{question_id}'
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching question: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Exception when fetching question: {e}")
        return None


def generate_exam(num_questions=5):
    """Generate an exam with a specified number of questions."""
    exam = []
    question_ids = random.sample(range(1, 490), num_questions)  # Randomly select question IDs

    for q_id in question_ids:
        question = fetch_question_from_api(q_id)
        if question:
            exam.append(question)
        else:
            print(f"Failed to fetch question with ID: {q_id}")

    print(exam)
    return exam


def main():
    exam = generate_exam()
    for i, question in enumerate(exam, 1):
        print(f"Question {i}: {question['Question']}")
        print(f"{question['Choice_A']}")
        print(f"{question['Choice_B']}")
        print(f"{question['Choice_C']}")
        print(f"{question['Choice_D']}")
        print(f"{question['Choice_E']}\n")


if __name__ == "__main__":
    main()
