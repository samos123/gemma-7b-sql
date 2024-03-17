from peft import LoraConfig, PeftModel
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
from datasets import load_dataset
import os
import time

model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

device = "cuda:0"

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

data = load_dataset("b-mc2/sql-create-context")
template = """Generate valid SQL query for a given natural language query and schema.
In your response only provide valid SQL for the schema, do not provide anything else.

Natural language query:
{question}

Schema:
```sql
{schema}
```

## RESPONSE
```sql
{answer}
```"""

def prompt(data):
    text = template.format(question=data["question"], schema=data["context"], answer=data["answer"])
    return {'text': text}

data = data.map(prompt)
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
)
print ("starting to fine-tune model")
start = time.time()
trainer.train()
print(f"finished training model in {time.time() - start} seconds")

text = """Generate valid SQL query for a given natural language query and schema.
In your response only provide valid SQL for the schema, do not provide anything else.

Natural language query:
Show me the users that have made a post on March 22, 2024 in the category AI

Schema:
```sql
CREATE TABLE users (
        user_id SERIAL PRIMARY KEY NOT NULL,
        name VARCHAR(256),
        email VARCHAR(256) UNIQUE NOT NULL
);
CREATE TABLE posts (
        post_id SERIAL PRIMARY KEY NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        title VARCHAR(256) NOT NULL,
        content TEXT,
        author_id INTEGER REFERENCES users(user_id)
);
CREATE TABLE profiles (
        profile_id SERIAL PRIMARY KEY NOT NULL,
        bio TEXT,
        user_id INTEGER NOT NULL REFERENCES users(user_id)
);
CREATE TABLE categories (
        category_id SERIAL PRIMARY KEY NOT NULL,
        name VARCHAR(256)
);
CREATE TABLE post_in_categories (
        post_id INTEGER NOT NULL REFERENCES posts(post_id),
        category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

## Response
"""

inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=800)
print("Verifying the model works by trying a prompt")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Merge with the base model to save a full model
new_model = '/tmp/gemma-7b-sql-peft'
trainer.model.save_pretrained(new_model)


base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

new_model_id = "samos123/gemma-7b-sql"

model.push_to_hub(new_model_id)

tokenizer.push_to_hub(new_model_id)