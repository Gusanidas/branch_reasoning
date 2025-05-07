import os
from dotenv import load_dotenv
from countdown_task import make_combined_countdown_tasks
from datasets import load_dataset, concatenate_datasets
import argparse

VERY_EASY_EXAMPLES = 0
EASY_EXAMPLES = 0
MEDIUM_EXAMPLES = 150
HARD_EXAMPLES = 300
VERY_HARD_EXAMPLES = 150

HF_DATASET_REPO_ID = "Gusanidas/countdown-tasks-dataset-harder"

def main():
    """Generates the dataset and pushes it to Hugging Face Hub."""
    parser = argparse.ArgumentParser(description='Generate and push dataset to Hugging Face Hub')
    parser.add_argument('--load-existing', action='store_true', help='Load and concatenate with existing dataset')
    args = parser.parse_args()

    print("Starting dataset generation...")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # Load existing dataset if requested
    existing_dataset = None
    if args.load_existing:
        print(f"Loading existing dataset from '{HF_DATASET_REPO_ID}'...")
        try:
            existing_dataset = load_dataset(HF_DATASET_REPO_ID)['train']
            print(f"Successfully loaded existing dataset with {len(existing_dataset)} examples.")
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
            return

    print(f"Generating new dataset with counts: "
          f"very_easy={VERY_EASY_EXAMPLES}, easy={EASY_EXAMPLES}, medium={MEDIUM_EXAMPLES}, "
          f"hard={HARD_EXAMPLES}, very_hard={VERY_HARD_EXAMPLES}")
    try:
        new_dataset = make_combined_countdown_tasks(
            very_easy=VERY_EASY_EXAMPLES,
            easy=EASY_EXAMPLES,
            medium=MEDIUM_EXAMPLES,
            hard=HARD_EXAMPLES,
            very_hard=VERY_HARD_EXAMPLES,
            shuffle_result=True
        )
        print(f"New dataset generated successfully with {len(new_dataset)} examples.")
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        return 

    # Combine datasets if loading existing
    if existing_dataset is not None:
        print("Concatenating existing and new datasets...")
        combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
        print(f"Combined dataset has {len(combined_dataset)} examples "
              f"({len(existing_dataset)} existing + {len(new_dataset)} new)")
    else:
        combined_dataset = new_dataset

    print(f"Attempting to push dataset to '{HF_DATASET_REPO_ID}'...")
    try:
        combined_dataset.push_to_hub(
            repo_id=HF_DATASET_REPO_ID,
            token=hf_token, 
            private=False 
        )
        print(f"Dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{HF_DATASET_REPO_ID}")
    except Exception as e:
        print(f"Error pushing dataset to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()