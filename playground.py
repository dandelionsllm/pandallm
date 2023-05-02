import os
import torch
from huggingface_hub import hf_hub_download

if __name__ == '__main__':
    from datasets import load_dataset

    data = load_dataset("natural_questions", 'default', beam_runner='DirectRunner')

    data = load_dataset('truthful_qa', 'multiple_choice')
