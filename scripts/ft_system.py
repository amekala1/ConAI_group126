# scripts/ft_system.py
 
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import ft_dataset_prep as prepare_ft_dataset
import ft_model_tuning as model_tuning
 
def load_ft_model():
    """
    Loads the fine-tuned GPT-2 model and tokenizer directly from Hugging Face Hub.
    """
    #prepare_ft_dataset.prepare_ft_dataset()

    #model_tuning.model_fine_tuning()

    repo_name = "amekala1/gpt2-finetuned-grp126"  # 🔹 change to your actual repo
 
    print(f"Loading fine-tuned GPT-2 model from Hugging Face Hub: {repo_name} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(repo_name)
    model = GPT2LMHeadModel.from_pretrained(repo_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Fine-tuned model loaded successfully from Hugging Face Hub.")
    return tokenizer, model
 
def ft_predict(query, tokenizer, model):
    """
    Generates a clean answer using the fine-tuned model.
    """
    prompt = f"Instruction: {query}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Use greedy decoding and stopping at EOS token
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,          # deterministic decoding
            repetition_penalty=2.0,   # reduce repeated tokens
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    try:
        answer = response.split("Response:")[1].strip()
    except IndexError:
        answer = "Sorry, I could not generate a clear response."
    confidence = 0.95  # placeholder
    return answer, confidence
 
 
def input_guardrail(query):
    """
    Simple input-side guardrail to filter relevant queries.
    """
    financial_keywords = [
        "revenue", "sales", "income", "profit", "assets", "liabilities",
        "cash flow", "dividend", "financial", "margin"
    ]
 
    is_relevant = any(keyword in query.lower() for keyword in financial_keywords)
 
    if is_relevant:
        return True, "Query is relevant."
    else:
        return False, "This chatbot is designed for financial questions only."
 
def run_ft_system(query, components):
    """
    Orchestrates the fine-tuned model pipeline for a given query.
    """
    tokenizer, model = components
 
    # Guardrail check
    is_relevant, guardrail_message = input_guardrail(query)
    if not is_relevant:
        return {
            "answer": guardrail_message,
            "confidence": 0,
            "response_time": 0,
            "is_relevant": False,
        }
 
    start_time = time.time()
 
    # Generate prediction
    answer, confidence = ft_predict(query, tokenizer, model)
 
    end_time = time.time()
    response_time = end_time - start_time
 
    return {
        "answer": answer,
        "confidence": confidence,
        "response_time": response_time,
        "is_relevant": True,
    }