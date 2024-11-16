import argparse

def generate_datasets(topic, data):
    search_term = f"""\"subject_name\":\"{topic}\""""

    def generate(d_type):
        with open(f"MedMCQA/{d_type}.json", 'r') as f:
            lines = f.readlines()
            filtered = [line.strip()+"," for line in lines if search_term in line]
            new_file = open(f"MedMCQA/{topic}_{d_type}.json", "w")
            new_file.write("[" + '\n'.join(filtered)[:-1] + "]")
    
    if data == 'train':
        generate('train')
    elif data == 'dev':
        generate('dev')
    elif data == 'test':
        generate('test')
    else:
        generate('train')
        generate('dev')
        generate('tes')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset for a specific topic in the MedMCQA dataset')
    parser.add_argument('-topic', type=str, help='topic for dataset generation')
    parser.add_argument('-data', type=str, default="all", help='train | dev | test | all')
    
    args = parser.parse_args()
    
    generate_datasets(args.topic, args.data)