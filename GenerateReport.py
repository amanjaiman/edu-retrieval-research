import pandas as pd
import matplotlib.pyplot as plt
import ace_tools_open as tools

class Report:
    def __init__(self, path):
        self.path = path
    
    def generate_report(self):
        df = pd.read_csv(self.path)
        answered_rows = df.loc[df['Generated Answer'].notnull()]
        unanswered_rows = df.loc[df['Generated Answer'].isnull()]
        correct = sum([row['Correct Answer'] == row['Generated Answer'] for i,row in answered_rows[['Correct Answer', 'Generated Answer']].iterrows()])

        self.total_questions = len(df)
        self.answered_questions = len(answered_rows)
        self.omissions = len(unanswered_rows)
        self.correct_questions = correct
        self.incorrect_questions = len(answered_rows) - correct

        self.accuracy = correct / len(answered_rows)
        self.avg_time = answered_rows.loc[:, 'Time Taken'].mean()

        self.calculate_calibration(answered_rows)

    def calculate_calibration(self, answered_rows):
        grouped = answered_rows.groupby('Confidence')
        statistics = []
        for confidence, group in grouped:
            num_rows = len(group)
            num_correct = (group["Correct Answer"] == group["Generated Answer"]).sum()
            num_incorrect = num_rows - num_correct
            percentage_correct = (num_correct / num_rows) * 100
            
            statistics.append({
                "Confidence": confidence,
                "Number of Rows": num_rows,
                "Number Correct": num_correct,
                "Number Incorrect": num_incorrect,
                "Percentage Correct": percentage_correct
            })

        self.calibration_stats_df = pd.DataFrame(statistics)

    def print_report(self):
        print('--------------------------------------')
        print('Total questions:', self.total_questions)
        print('Answered questions:', self.answered_questions)
        print('Correct questions:', self.correct_questions)
        print('Incorrect questions:', self.incorrect_questions)
        
        print('--------------------------------------')
        print('Accuracy:', self.accuracy)
        print('Omissions:', self.omissions)
        print('Average time (for answered questions):', self.avg_time)

        print('--------------------------------------')
        tools.display_dataframe_to_user(name="Calibration", dataframe=self.calibration_stats_df)
        
        # Prepare data points for graphing
        x = self.calibration_stats_df["Confidence"]
        y = self.calibration_stats_df["Percentage Correct"]

        # Plot the graph
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', label="Accuracy by Confidence")
        plt.title("Model Accuracy by Confidence Level")
        plt.xlabel("Confidence")
        plt.ylabel("Percentage Correct")
        plt.grid(True)
        plt.legend()
        plt.show()


def main(path):
    report = Report(path)
    report.generate_report()
    report.print_report()