# scripts/ft_model_tuning.py

import json
import os
import torch
import numpy as np
import shutil
import time
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import HfApi, login, create_repo
from sklearn.metrics import accuracy_score
import streamlit as st

def compute_metrics(eval_pred):
    """
    A custom function to compute accuracy during evaluation.
    This metric is more suitable for text generation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def get_baseline_predictions(model, tokenizer, test_questions):
    """
    Generates predictions from the base model before fine-tuning.
    This serves as your baseline benchmark.
    """
    prompts = [f"Instruction: {q}\nResponse:" for q in test_questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # deterministic
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def push_to_huggingface_hub(model, tokenizer, repo_name, hf_token):
    """
    Push the fine-tuned model and tokenizer to Hugging Face Hub
    """
    try:
        # Login to Hugging Face
        login(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, token=hf_token, exist_ok=True)
        except Exception as e:
            st.warning(f"Repo creation note: {e}")
        
        # Push model and tokenizer to Hub
        model.push_to_hub(repo_name, use_temp_dir=False)
        tokenizer.push_to_hub(repo_name, use_temp_dir=False)
        
        st.success(f"✅ Model successfully pushed to Hugging Face Hub: {repo_name}")
        return True
        
    except Exception as e:
        st.error(f"❌ Error pushing to Hugging Face Hub: {e}")
        return False

def model_fine_tuning():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    # 3.2: Model Selection - Using GPT-2 for a generative task
    model_name = "gpt2"
    model_dir = os.path.join(project_root, "models", "gpt2")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
    except OSError:
        st.info("Local model not found. Downloading from Hugging Face...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    # Temporary directory for trainer output
    output_temp_dir = os.path.join(project_root, "output_temp", f"tmp_trainer_{int(time.time())}")

    # --- Load tokenizer and model ---
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
    except OSError:
        st.info("Local model not found. Downloading from Hugging Face...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    # Load fine-tuning dataset
    ft_dataset_path = os.path.join(project_root, "data", "ft_dataset.json")
    dataset = load_dataset('json', data_files=ft_dataset_path, split='train')

    # Safe dataset slicing
    train_end = min(40, len(dataset))
    test_end = min(50, len(dataset))
    dataset_shuffled = dataset.shuffle(seed=42)
    train_dataset = dataset_shuffled.select(range(train_end))
    test_dataset = dataset_shuffled.select(range(train_end, test_end))

    # Tokenize dataset
    def tokenize_function(examples):
        text = [
            f"Instruction: {instr}\nResponse: {resp}{tokenizer.eos_token}"
            for instr, resp in zip(examples['instruction'], examples['response'])
        ]
        return tokenizer(text, truncation=True, padding="max_length", max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "response"])
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "response"])

    # 3.3: Baseline Benchmarking
    st.write("--- Baseline Benchmarking (Pre-Fine-Tuning) ---")
    test_questions = [q['instruction'] for q in test_dataset]
    baseline_predictions = get_baseline_predictions(model, tokenizer, test_questions)
    for q, a in zip(test_questions, baseline_predictions):
        st.write(f"Question: {q}\nBaseline Response: {a.strip()}")
        st.write("-" * 20)

    # 3.4: Fine-Tuning Setup
    hyperparameters = {
        "learning_rate": 5e-5,
        "batch_size": 2,
        "num_epochs": 3,
        "compute_setup": "GPU" if torch.cuda.is_available() else "CPU"
    }

    os.makedirs(output_temp_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_temp_dir,
        overwrite_output_dir=True,
        num_train_epochs=hyperparameters["num_epochs"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        learning_rate=hyperparameters["learning_rate"],
        save_strategy="epoch",
        logging_dir=os.path.join(output_temp_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 3.4 & 3.5: Fine-Tuning
    st.write("\n--- Starting Supervised Instruction Fine-Tuning ---")
    trainer.train()

    # Save final fine-tuned model locally
    final_ft_model_dir = os.path.join(project_root, "models", "gpt2-finetuned")
    #os.makedirs(final_ft_model_dir, exist_ok=True)
    #st.write(f"\nSaving Fine-tuned model locally")
    #trainer.save_model(final_ft_model_dir)
    final_model = trainer.model
    #st.write(f"\nFine-tuned model saved locally to: {final_ft_model_dir}")

    # Push to Hugging Face Hub automatically
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))
    repo_name = "amekala1/gpt2-finetuned"  # Change this to your desired repo name

    if hf_token:
        with st.spinner("Pushing model to Hugging Face Hub..."):
            success = push_to_huggingface_hub(final_model, tokenizer, repo_name, hf_token)
            if success:
                st.success(f"Model available at: https://huggingface.co/{repo_name}")
    else:
        st.warning("Hugging Face token not found. Model not pushed to Hub.")

    # Cleanup temporary directory
    if os.path.exists(output_temp_dir):
        shutil.rmtree(output_temp_dir)
        st.write(f"Temporary directory '{output_temp_dir}' cleaned up.")

    # Post-Fine-Tuning Evaluation
    st.write("\n--- Post-Fine-Tuning Evaluation ---")
    results = trainer.evaluate()
    st.write(results)