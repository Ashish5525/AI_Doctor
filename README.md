# AI_Doctor

Fine-Tuning Large Language Models (LLMs) with Hugging Face for Medical Information

## Overview

AI Doctor is a project focused on fine-tuning a pre-trained Llama2 model to generate accurate and detailed medical information. By leveraging the powerful tools provided by Hugging Face, we aim to create a model that can assist users with medical queries, providing comprehensive explanations and information on various medical terms and conditions.

## Dataset

The dataset used for this project is sourced from [Hugging Face](https://huggingface.co/datasets/gamino/wiki_medical_terms). This dataset consists of medical terms and their descriptions formatted for easy ingestion by LLMs. It includes a diverse range of medical topics and terms, making it ideal for training a model intended to provide detailed medical information.

## Goals

- Fine-tune a pre-trained Llama2 model to understand and generate medical information.
- Ensure the model can handle a wide range of medical queries.
- Utilize state-of-the-art tools from Hugging Face to achieve optimal training and performance.

## Methodology

1. **Data Preparation:** Utilize the wiki_medical_terms dataset from Hugging Face, which contains a wealth of medical terms and descriptions.
2. **Model Selection:** Start with a pre-trained Llama2 model, which offers a robust foundation for further fine-tuning.
3. **Fine-Tuning:** Fine-tune the Llama2 model using supervised learning techniques, focusing on improving the model's ability to understand and generate medical content.
4. **Evaluation:** Continuously evaluate the model's performance to ensure accuracy and relevance of the generated information.
   
## Tools and Libraries

- **Hugging Face Transformers:** For accessing and fine-tuning the Llama2 model.
- **Datasets:** For loading and processing the medical terms dataset.
- **Training Libraries:** Utilizing libraries like SFTTrainer and peft for supervised fine-tuning.
- **BitsAndBytes:** For efficient model quantization and computation.

## Installation

To get started, you need to install the required libraries. You can install them using the following commands:
 ```bash
 pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
pip install huggingface_hub
 ```

## Usage

### Step 1: Importing Libraries
 ```python
import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
 ```

### Step 2: Loading the Model
 ```python
llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v2", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, "float16"), bnb_4bit_quant_type="nf4"))
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

 ```
### Step 3: Loading the Tokenizer
 ```python
llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v2", trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"
 ```

### Step 4: Setting the training arguments
 ```python
training_arguments = TrainingArguments(output_dir="./results", per_device_train_batch_size=4, max_steps=100)
 ```

### Step 5: Creating the Supervised Fine-Tuning Trainer
 ```python
llama_sft_trainer = SFTTrainer(model=llama_model,
                               args=training_arguments,
                               train_dataset=load_dataset(path="gamino/wiki_medical_terms", split="train"),
                               tokenizer=llama_tokenizer,
                               peft_config=LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1),
                               dataset_text_field="text")

 ```

### Step 6: Training The Model
 ```python
llama_sft_trainer.train()
 ```

### Step 7: Chatting with the Model
 ```python
user_prompt = "Please tell me about Bursitis"
text_generation_pipeline = pipeline(task="text-generation", model=llama_model, tokenizer=llama_tokenizer, max_length=300)
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])
 ```

## Example

A typical use case involves inputting a medical query into the model, such as "Please tell me about Bursitis". The model will then generate a response that provides detailed information about the condition.
