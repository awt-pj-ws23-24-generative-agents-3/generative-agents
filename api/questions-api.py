from flask import Flask, jsonify
import pandas as pd

app = Flask("Questions API")

# Load and prepare the CSV data
csv_data = pd.read_csv('mc-dataset.csv')
csv_data.columns = [col.strip() for col in csv_data.columns]  # rm space from column names

# Convert DataFrame to a list of dictionaries, including the ID
questions_list = csv_data.to_dict(orient='records')

# Create a dictionary with IDs as keys and question data dictionaries as values
questions_dict = {question['ID']: question for question in questions_list}


@app.route('/questions/<int:question_id>', methods=['GET'])
def get_question(question_id):
    question = questions_dict.get(question_id)
    if question:
        return jsonify(question)
    else:
        return jsonify({"error": "Question not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
