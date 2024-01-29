from flask import Flask, jsonify
import pandas as pd

app = Flask("Questions API")

# Load and prepare the CSV data
csv_data = pd.read_csv('../data/mc-dataset.csv')
csv_data.columns = [col.strip() for col in csv_data.columns]  # rm space from column names
questions_dict = csv_data.set_index('ID').T.to_dict()



@app.route('/questions/<int:question_id>', methods=['GET'])
def get_question(question_id):
    question = questions_dict.get(question_id)
    if question:
        return jsonify(question)
    else:
        return jsonify({"error": "Question not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, port=8000)
