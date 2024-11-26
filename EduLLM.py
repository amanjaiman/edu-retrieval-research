import argparse
import datetime
import json
import random
import time

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import pandas as pd
from tqdm import tqdm

from helpers import Answer
from prompt import create_prompt_with_answer_prompt

class ModelWithNoRetrieval:
    def run(self, questions_file_path, num_questions):
        correct_answers = {
            1: "A",
            2: "B",
            3: "C",
            4: "D"
        }

        df = pd.DataFrame(columns=['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Generated Answer', 'Generated Explanation', 'Confidence', 'Time Taken'])
        
        with open(questions_file_path, 'r') as f:
            questions = json.loads(f.read())
            #random.shuffle(questions)

            for i in tqdm(range(num_questions)):
                start_time = time.time()

                q_json = questions[i]
                question_and_answer_prompt = create_prompt_with_answer_prompt(q_json)

                #print(f'Calling model for question {i+1}...')
                result = self.answer_llm.invoke(question_and_answer_prompt)
                #print(result)

                end_time = time.time()

                if result:
                    question_entry = {
                        "Question": q_json['question'],
                        "Option A": q_json['opa'],
                        "Option B": q_json['opb'],
                        "Option C": q_json['opc'],
                        "Option D": q_json['opd'],
                        "Correct Answer": correct_answers[q_json['cop']],
                        "Generated Answer": result.correct_answer,
                        "Generated Explanation": result.explanation,
                        "Confidence": result.confidence.split('%')[0],
                        "Time Taken": end_time - start_time
                    }
                else:
                    question_entry = {
                        "Question": q_json['question'],
                        "Option A": q_json['opa'],
                        "Option B": q_json['opb'],
                        "Option C": q_json['opc'],
                        "Option D": q_json['opd'],
                        "Correct Answer": correct_answers[q_json['cop']],
                        "Generated Answer": "N/A",
                        "Generated Explanation": "N/A",
                        "Confidence": "N/A",
                        "Time Taken": end_time - start_time
                    }
                
                row_df = pd.DataFrame([question_entry])
                df = pd.concat([df, row_df], ignore_index=True)
                
                time.sleep(1.5)
        
        date = datetime.datetime.now()
        filename = f"{questions_file_path.split('/')[1].split('.')[0]}_{num_questions}_{self.model_name}_{date.year}{date.month}{date.day}"
        df.to_csv(f'outputs/no_rag/{filename}.csv')

    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'gpt4o':
            self.answer_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Answer)
        elif model_name == 'claude3.5':
            self.answer_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620").with_structured_output(Answer)
        else:
            self.answer_llm = ChatOllama(model=model_name).with_structured_output(Answer)

def main(model, qcount, path):
    model = ModelWithNoRetrieval(model)
    model.run(path, qcount)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run basic LLM")
#     parser.add_argument('--model', type=str, help='specify the model to use: gpt4o | claude3.5 | llama3.2 (local using Ollama)')
#     parser.add_argument('--qcount', type=int, help='the number of questions to run')
#     parser.add_argument('--path', type=str, help='the path to the question json')

#     args = parser.parse_args()

#     model = ModelWithNoRetrieval(args.model)
#     model.run(args.path, args.qcount)
