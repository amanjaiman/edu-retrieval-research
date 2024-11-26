import argparse
import datetime
import json
import random
import time

import pandas as pd

from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from helpers import Answer
from KGNodeIndex import WordSimilarityIndex
from prompt import QA_TEMPLATE, create_prompt_with_answer_prompt, create_question_prompt, generate_cypher_template

qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=QA_TEMPLATE
)

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

class ModelWithKG:
    def maybe_extract_cypher_query(self, formatted_string):
        # Remove the triple quotes and the surrounding code block markers
        start_marker = "```cypher\n"
        end_marker = "\n```"
        
        # Find the start and end of the actual query
        start_index = formatted_string.find(start_marker) + len(start_marker)
        end_index = formatted_string.find(end_marker, start_index)
        
        # Extract the query from the string
        if start_index != -1 and end_index != -1:
            return formatted_string[start_index:end_index].strip()
        else:
            # Return None or an empty string if the markers are not found
            return ""
    
    def generate_query_with_retry(self, question, known_nodes, retries=3):
        messages = [
            SystemMessage(generate_cypher_template(self.graph.get_schema, known_nodes)),
        ]
        
        chat_history = InMemoryChatMessageHistory(messages=messages)
        def get_session_history(session_id):
            if session_id != "1":
                return InMemoryChatMessageHistory()
            return chat_history

        graph_chain_with_history = RunnableWithMessageHistory(self.graph_llm, get_session_history)

        config = {"configurable": {"session_id": "1"}}
        current_try = 0
        human_message = question

        while current_try < retries:
            #print(f"{current_try}: {human_message}")
            ai_response = graph_chain_with_history.invoke({"input": human_message}, config=config)
            messages.append(HumanMessage(human_message))
            messages.append(ai_response)

            generated_query = self.maybe_extract_cypher_query(ai_response.content)
            #print(generated_query)
            try:
                context = self.graph.query(generated_query)

                if len(context) != 0:
                    return (context, current_try+1)
                else:
                    human_message = f"This generated query returned no results. Try again with a different query. If needed, make the query less specific." if current_try == retries else ""
            except Exception as err:
                human_message = f"Using this generated query to query the knowledge graph resulted in the following error: {err}"
            
            current_try += 1

        return ([], current_try)
    
    def run(self, questions_file_path, num_questions, retry_per_question):
        correct_answers = {
            1: "A",
            2: "B",
            3: "C",
            4: "D"
        }
        df = pd.DataFrame(columns=['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Generated Answer', 'Generated Explanation', 'Confidence', 'KG Context', 'Tries Used', 'Time Taken'])
        
        with open(questions_file_path, 'r') as f:
            questions = json.loads(f.read())
            #random.shuffle(questions)

            for i in tqdm(range(num_questions)):
                start_time = time.time()

                q_json = questions[i]
                question_prompt = create_question_prompt(q_json)
                question_and_answer_prompt = create_prompt_with_answer_prompt(q_json)

                known_nodes = []
                if self.node_index:
                    known_nodes = self.node_index.search_similar_words(question_prompt)

                #print(f'Calling model for question {i+1}...')
                (context, tries) = self.generate_query_with_retry(question_prompt, known_nodes, retry_per_question)

                result = self.answer_chain.invoke({"question": question_and_answer_prompt, "context": context})
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
                        "KG Context": context,
                        "Tries Used": tries,
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
                        "KG Context": context,
                        "Tries Used": tries,
                        "Time Taken": end_time - start_time
                    }

                row_df = pd.DataFrame([question_entry])
                df = pd.concat([df, row_df], ignore_index=True)
        
        date = datetime.datetime.now()
        filename = f"{questions_file_path.split('/')[1].split('.')[0]}_{num_questions}_{self.model_name}_{retry_per_question}_{date.year}{date.month}{date.day}"
        df.to_csv(f'outputs/kg_rag/{filename}.csv')
    
    def __init__(self, model_name, with_index=False):
        self.model_name = model_name
        self.graph = Neo4jGraph()

        if model_name == 'gpt4o':
            self.graph_llm = ChatOpenAI(model="gpt-4o")
            self.answer_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Answer)
        elif model_name == 'claude3.5':
            self.graph_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
            self.answer_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620").with_structured_output(Answer)
        else:
            self.graph_llm = ChatOllama(model=model_name)
            self.answer_llm = ChatOllama(model=model_name).with_structured_output(Answer)

        self.answer_chain = qa_prompt | self.answer_llm

        self.node_index = None
        if with_index:
            self.node_index = WordSimilarityIndex()
            self.node_index.load_index()

def main(model, qcount, path, retries, with_index):
    model = ModelWithKG(model, with_index)
    model.run(path, qcount, retries)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run KG RAG model")
#     parser.add_argument('--model', type=str, help='specify the model to use: gpt4o | claude3.5 | llama3.2 (local using Ollama)')
#     parser.add_argument('--qcount', type=int, help='the number of questions to run')
#     parser.add_argument('--path', type=str, help='the path to the question json')
#     parser.add_argument('--retries', type=int, default=3, help='the number of times to retry context retrival')
#     parser.add_argument('--with_index', type=bool, default=False, help='to run with index over the knowledge graph nodes')

#     args = parser.parse_args()

#     model = ModelWithKG(args.model, args.with_index)
#     model.run(args.path, args.qcount, args.retries)

# python kg_rag.py --path=MedMCQA/Biochemistry_dev.json --qcount=1 --retries=3
# python kg_rag.py --path=MedMCQA/Biochemistry_dev.json --qcount=1 --retries=3 --index_path=kg_node_index.bin --words_path=words.pkl