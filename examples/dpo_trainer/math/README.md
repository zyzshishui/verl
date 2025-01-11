# DPO Trainer Example
## Model Selection
We use [QWen2.5-7b-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as demo
## Dataset
OpenAI GSM8k
## Data Generation

For each prompt in the gsm8k dataset, we use QWen2.5-7b-instruct to generate 5 answers. We extract the answers via rules. If the answer is correct, we give it 1 point. If the answer is incorrect, but the answer follow the instruction, we give it 0.1 point. Otherwise, we give it 0 point. We rank the responses for each prompt.

## DPO training