from datasets import load_dataset
import random
import argparse
import os
import csv

if __name__ == "__main__":
    def t(example):
        i = random.random()
        if i < 0.5:
            example['option_A'] = example['targets']['choices'][0]
            example['option_B'] = example['targets']['choices'][1]
            example['answer'] = "A"
        else:
            example['option_A'] = example['targets']['choices'][1]
            example['option_B'] = example['targets']['choices'][0]
            example['answer'] = "B"
        return example
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="hhh_data")
    parser.add_argument("--n_shots", "-n", type=int, default=4)
    args = parser.parse_args()

    harmless = load_dataset("HuggingFaceH4/hhh_alignment", "harmless")['test']
    helpful = load_dataset("HuggingFaceH4/hhh_alignment", "helpful")['test']
    honest = load_dataset("HuggingFaceH4/hhh_alignment", "honest")['test']

    harmless = harmless.map(t)
    helpful = helpful.map(t)
    honest = honest.map(t)

    harmless = harmless.remove_columns(['targets'])
    helpful = helpful.remove_columns(['targets'])
    honest = honest.remove_columns(['targets'])

    if args.n_shots > 0:
        harmless = harmless.train_test_split(args.n_shots)
        helpful = helpful.train_test_split(args.n_shots)
        honest = honest.train_test_split(args.n_shots)
    
    harmless_dev = harmless['test']
    harmless_test = harmless['train']
    helpful_dev = helpful['test']
    helpful_test = helpful['train']
    honest_dev = honest['test']
    honest_test = honest['train']




    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "dev")):
        os.makedirs(os.path.join(args.save_dir, "dev"))
    if not os.path.exists(os.path.join(args.save_dir, "test")):
        os.makedirs(os.path.join(args.save_dir, "test"))

    harmless_dev.to_csv(os.path.join(args.save_dir,"dev", "harmless_dev.csv"), header=False, sep=",",quotechar='"',index=False, quoting=csv.QUOTE_ALL)
    harmless_test.to_csv(os.path.join(args.save_dir,"test", "harmless_test.csv"), header=False, sep=",",quotechar='"',index=False, quoting=csv.QUOTE_ALL)
    helpful_dev.to_csv(os.path.join(args.save_dir,"dev", "helpful_dev.csv"), header=False, sep=",",quotechar='"',index=False, quoting=csv.QUOTE_ALL)
    helpful_test.to_csv(os.path.join(args.save_dir,"test", "helpful_test.csv"), header=False, sep=",",quotechar='"',index=False, quoting=csv.QUOTE_ALL)
    honest_dev.to_csv(os.path.join(args.save_dir,"dev", "honest_dev.csv"), header=False, sep=",",quotechar='"',index=False, quoting=csv.QUOTE_ALL)
    honest_test.to_csv(os.path.join(args.save_dir,"test", "honest_test.csv"), header=False, sep=",",quotechar='"',index=False, quoting=csv.QUOTE_ALL)

