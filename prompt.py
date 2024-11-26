def create_prompt_with_answer_prompt(question, context=""):
    question_text = question['question']
    opa = question['opa']
    opb = question['opb']
    opc = question['opc']
    opd = question['opd']

    question_prompt = f"""Question: {question_text}
A) {opa}
B) {opb}
C) {opc}
D) {opd}

Answer: among the answer choices A, B, C, and D, the answer is """
    
    if context != "":
        return f"""Context: {context}\n\n{question_prompt}"""
    else:
        return question_prompt
    
def create_question_prompt(question):
    question_text = question['question']
    opa = question['opa']
    opb = question['opb']
    opc = question['opc']
    opd = question['opd']

    question_prompt = f"""Question: {question_text}
A) {opa}
B) {opb}
C) {opc}
D) {opd}"""
    
    return question_prompt

def generate_cypher_template(schema, known_nodes=[]):
    schema = schema.replace("{", "(")
    schema = schema.replace("}", ")")

    formatted_nodes = '\n'.join([f"id: {node}, label: {label}\n" for (node,label) in known_nodes])
    known_nodes_prompt = f"""\nHere are the nodes that may help to answer the question:
    {formatted_nodes}\n""" if len(known_nodes) != 0 else ""

    return f"""
You are working with the following Neo4j schema: {schema}
{known_nodes_prompt}
You are a cypher query generator. Given a user question and answer choices and the list of nodes above, your job is to generate a cypher query using the relevant nodes and their nearby connections up to depth 2.
The goal is to provide some amount of context for the question. The cypher query should have the correct syntax so that your output can be used to query the Neo4j graph.
Don't make the query too specific, as that could lead to no results when querying the graph.

Cypher Query:
"""

CYPHER_GENERATION_TEMPLATE = """
You are working with the following Neo4j schema: {schema}

Given a user question and answer choices, generate a Cypher query to find relevant nodes for the question and answer choices and their nearby connections up to depth 2.
You can use wildcards like MATCH (n)-[r]-(m) to ensure we retrieve related nodes.
The goal is to provide some amount of context for the question.

Do not include any preamble or explanation. Start and end your response with the cypher query so that the output can be used to query the Neo4j graph.
"""

QA_TEMPLATE = """
You are to answer the following question.
{question}

You are also given the following context:
{context}

Answer: among the answer choices A, B, C, and D, the answer is 
"""