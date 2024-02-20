import csv
import argparse
from datasets import load_dataset

def format_chatgpt_prompts(args):
    fout = open(args.output_path, "w")
    args.data_path = "MohamedRashad/ChatGPT-prompts"
    dataset = load_dataset(args.data_path)
    i = -1
    for row in dataset["train"]:
        fout.write(row["human_prompt"] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    format_chatgpt_prompts(args)