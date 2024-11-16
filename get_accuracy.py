import argparse

import pandas as pd

def get_accuracy(path):
    df = pd.read_csv(path)
    correct = sum([row['Correct Answer'] == row['Generated Answer'] for i,row in df[['Correct Answer', 'Generated Answer']].iterrows()])
    unanswered = df['Generated Answer'].value_counts().get('N/A', 0)
    return {
        "accuracy": correct / df.shape[0],
        "unanswered": unanswered
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate accuracy of outputs")
    parser.add_argument('--path', type=str, help='the path to the question json')

    args = parser.parse_args()

    accuracy = get_accuracy(args.path)
    print(accuracy)