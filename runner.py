import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    parser = argparse.ArgumentParser(description="Run different functionalities")
    subparsers = parser.add_subparsers(dest="command", help="Choose a functionality to run")

    parser_build_kg = subparsers.add_parser("build_kg", help="Build knowledge graph")
    parser_build_kg.add_argument('--dir_path', help="Path to the directory with PDFs to build KG on")
    parser_build_kg.add_argument('--chunk_size', type=int, help="Chunk size to build KG")

    parser_build_index = subparsers.add_parser("build_index", help="Build index retriever")
    parser_build_index.add_argument('--dir_path', help="Path to the directory with PDFs to build the index retriever on")
    parser_build_index.add_argument('--chunk_size', type=int, help="Chunk size to build index retriever")
    
    subparsers.add_parser("build_node_index", help="Build index on KG nodes")

    parser_kg_rag = subparsers.add_parser("kg_rag", help="Run KG Rag functionality")
    parser_kg_rag.add_argument('--model', type=str, help='specify the model to use: gpt4o | claude3.5 | <OTHER> (local using Ollama, ex: llama3.2)')
    parser_kg_rag.add_argument('--qcount', type=int, help='the number of questions to run')
    parser_kg_rag.add_argument('--path', type=str, help='the path to the question json')
    parser_kg_rag.add_argument('--retries', type=int, default=3, help='the number of times to retry context retrival')
    parser_kg_rag.add_argument('--with_index', type=bool, default=False, help='to run with index over the knowledge graph nodes')
    
    parser_index_llm = subparsers.add_parser("index_rag", help="Run basic LLM functionality")
    parser_index_llm.add_argument('--model', type=str, help='specify the model to use: gpt4o | claude3.5 | <OTHER> (local using Ollama, ex: llama3.2)')
    parser_index_llm.add_argument('--qcount', type=int, help='the number of questions to run')
    parser_index_llm.add_argument('--path', type=str, help='the path to the question json')

    parser_llm = subparsers.add_parser("basic_llm", help="Run basic LLM functionality")
    parser_llm.add_argument('--model', type=str, help='specify the model to use: gpt4o | claude3.5 | <OTHER> (local using Ollama, ex: llama3.2)')
    parser_llm.add_argument('--qcount', type=int, help='the number of questions to run')
    parser_llm.add_argument('--path', type=str, help='the path to the question json')

    report = subparsers.add_parser("generate_report", help="Generate the report for the output")
    report.add_argument('--path', type=str, help="the path to the output csv")

    args = parser.parse_args()

    if args.command == "kg_rag":
        import EduKGRag
        EduKGRag.main(args.model, args.qcount, args.path, args.retries, args.with_index)
    elif args.command == "index_rag":
        import EduIndexRag
        EduIndexRag.main(args.model, args.qcount, args.path)
    elif args.command == "basic_llm":
        import EduLLM
        EduLLM.main(args.model, args.qcount, args.path)
    elif args.command == "build_node_index":
        import KGNodeIndex
        KGNodeIndex.build_index()
    elif args.command == "build_kg":
        import BuildKG
        BuildKG.main(args.dir_path, args.chunk_size)
    elif args.command == "build_index":
        import BuildIndexRetriever
        BuildIndexRetriever.main(args.dir_path, args.chunk_size)
    elif args.command == "generate_report":
        import GenerateReport
        GenerateReport.main(args.path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

# python runner.py build_index --dir_path=BiochemData/ --chunk_size=512
# python runner.py build_kg --dir_path=BiochemData/ --chunk_size=512
# python runner.py build_node_index

# python runner.py basic_llm --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json
# python runner.py index_rag --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json

# python runner.py kg_rag --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json --retries=3
# python runner.py kg_rag --model=<MODEL> --qcount=<COUNT> --path=MedMCQA/Biochemistry_dev.json --retries=3 --with_index

# python runner.py generate_report --path=<PATH>