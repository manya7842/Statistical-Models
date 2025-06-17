import torch 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from accelerate import PartialState

device_string = PartialState().process_index
device_map={'':device_string}
# Load dataset
fair_responses = load_dataset("respai-lab/fair_responses", split="train")
formatted_data=[]
for item in fair_responses:
    formatted_data.append({
        "text": f"""### Question: {item['prompt']}\n ### Answer: {item['response']}"""
    })

train_data = Dataset.from_list(formatted_data)
# Model and tokenizer setup
model_id = "Qwen/Qwen2.5-7B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen doesn't have a pad token by default
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="cache_dir",
    torch_dtype=torch.bfloat16,
    #device_map="auto",
    trust_remote_code=True,
    # device_map=device_map, 
    use_cache=False,
    attn_implementation="flash_attention_2"
)

# Tokenization function
def tokenize_function(examples):
    assert isinstance(examples["prompt"], list), "Expected list of prompts"

    

    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="longest",
        # return_special_tokens_mask=True,
        return_tensors="pt"
    )

# Apply tokenization to the dataset
tokenized_dataset = train_data.map(
    lambda x : tokenizer(x['text'], padding='longest', truncation=True),
    batched=True,
    # num_proc=4,
    remove_columns=train_data.column_names
)
# print(tokenized_dataset)
# Training arguments
response_template = " ### Answer:"
instruction_template = "### Question:"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, instruction_template=instruction_template, tokenizer=tokenizer)
training_args = SFTConfig(
    max_length=512,
    output_dir="./SFTtmp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    logging_steps=10,
    weight_decay=0.1, 
    save_steps=50,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    learning_rate=6e-5,
    # dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant" : False}
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=collator
)
# Start training
trainer.train()