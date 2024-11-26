import datetime
import json
import time

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import pandas as pd
from tqdm import tqdm
from helpers import Answer
from prompt import create_question_prompt

template = """Context:
{context}

{question}

{answer_prompt}"""

class ModelWithIndex:
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
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

            for i in tqdm(range(num_questions)):
                start_time = time.time()

                q_json = questions[i]
                question_prompt = create_question_prompt(q_json)

                rag_chain = (
                    {"context": self.retriever | self.format_docs, "question": lambda x: question_prompt, "answer_prompt": RunnablePassthrough()}
                    | self.prompt
                    | self.llm
                )
                result = rag_chain.invoke("Answer: among the answer choices A, B, C, and D, the answer is ")

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

                date = datetime.datetime.now()
                filename = f"{questions_file_path.split('/')[1].split('.')[0]}_{num_questions}_{self.model_name}_{date.year}{date.month}{date.day}"
                df.to_csv(f'outputs/index_rag/{filename}.csv')

    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'gpt4o':
            self.llm = ChatOpenAI(model="gpt-4o").with_structured_output(Answer)
        elif model_name == 'claude3.5':
            self.llm = ChatAnthropic(model="claude-3-5-sonnet-20240620").with_structured_output(Answer)
        else:
            self.llm = ChatOllama(model=model_name).with_structured_output(Answer)

        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.load_local(
            "data_index", embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        self.prompt = PromptTemplate.from_template(template)

def main(model, qcount, path):
    model = ModelWithIndex(model)
    model.run(path, qcount)