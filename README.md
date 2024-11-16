# (WIP) edu-retrieval-research
 
## Problem Statement:
Large language models (LLMs) have made significant progress in understanding and generating human-like text, but they still face challenges with accuracy and consistency, especially in specialized domains like education. In education, LLMs have the potential to greatly benefit teachers and students, but hallucinations can lead to significant negative outcomes by conveying incorrect information or providing misleading answers. This research aims to explore the use of knowledge graphs (KGs) to reduce hallucinations specifically within the education domain.

## Setup:
To use Neo4j, you'll need to set the following environment variables:
* `NEO4J_PASSWORD`
* `NEO4J_URI`
* `NEO4J_USERNAME`

Next, setup whichever LLM you want to work with:
* If using gpt4o, set the `OPENAI_API_KEY` environment variable
* If using claude-3.5, set the `ANTHROPIC_API_KEY` environment variable
* If using an open source model, download Ollama and run whichever model you want.

## Commands:
#### Retrieval setup
* To build the index retriver from the BiochemData:
  > `python runner.py build_index --dir_path=BiochemData/ --chunk_size=512`
* To build the Neo4j knowledge graph from the BiochemData:
  > `python runner.py build_kg --dir_path=BiochemData/ --chunk_size=512`
* T0 build the index on the Neo4j graph nodes:
  > `python runner.py build_node_index`

#### Models
* Running the basic LLM with no retrieval:
  > `python runner.py basic_llm --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json`
* Running the LLM with index retrieval:
  > `python runner.py index_rag --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json`
* Running the LLM with knowledge graph retrieval:
  > `python runner.py kg_rag --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json --retries=3`
* Running the LLM with knowledge graph retrieval utilizing the node index:
  > `python runner.py kg_rag --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json --retries=3 --with_index=True`

Options:
* `model`: The model can be either `gpt4o`, `claude3.5`, or whichever model you are using with Ollama (for example `llama3.2`)
* `qcount`: The number of questions to answer
* `path`: The path to the questions
* `retries`: How many times the model will try to generate a valid query for the knowledge graph (only for kg_rag)
* `with_index`: Whether or not to use the node index to generate queries for the knwoledge graph (only for kg_rag)
