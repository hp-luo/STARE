
"""
Preprocess the DAPO-Math-17k dataset to parquet format
"""
import os
import datasets
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/retool")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    dapo_dataset = datasets.load_dataset(
        "haizhongzheng/DAPO-Math-17K-cleaned"
    )

    dataset = dapo_dataset["train"]
    train_dataset = dataset.shuffle(seed=42).select(range(0, len(dataset)))

    def process_dapo(example, idx):
        prompt = example.get("prompt", "")
        user_question = prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        messages = [{'content': user_question, 'role': 'user'}]
        ground_truth = example.get("target", "")
        ability = example.get("ability", "")

        data = {
            "data_source": "Short_CoT",
            "prompt": messages,
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                'answer': ground_truth,
                'question': user_question,
                'source': "Short_CoT"
            },
        }
        return data

    processed_train = train_dataset.map(function=process_dapo, with_indices=True)
    local_dir = "recipe/STARE/datasets/train"

    os.makedirs(local_dir, exist_ok=True)
    processed_train.to_parquet(os.path.join(local_dir, "dapo_17k_text_train.parquet"))


