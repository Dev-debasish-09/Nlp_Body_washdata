
# SmolLM-1.7B Bodywash Tagging Model

A fine-tuned language model for predicting Level I factors (tags) for body wash products using QLoRA efficient fine-tuning.

## Model Details

- **Base Model:** [HuggingFaceTB/SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)
- **Fine-tuning Method:** QLoRA (Low-Rank Adaptation)
- **Library:** PEFT 0.17.1
- **Task:** Text-to-Text Generation for Product Tagging
- **Input Format:** Body wash product descriptions
- **Output Format:** Comma-separated Level I factors/tags

## Training Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | 1,729,464,320 |
| Trainable Parameters | 18,087,936 |
| Trainable % | 1.05% |
| Training Hardware | Google Colab T4 GPU |
| Final Training Loss | 1.1208 |
| Final Validation Loss | 0.8655 |

## Training Performance

The model showed excellent convergence during training with consistent decrease in both training and validation loss, indicating good generalization without overfitting.

## How to Use

### Installation

```bash
pip install transformers peft torch accelerate
```

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load configuration
config = PeftConfig.from_pretrained("your-username/your-model-name")

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "your-username/your-model-name")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Set padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Inference

```python
def predict_bodywash_tags(product_description):
    """
    Predict Level I factors for a body wash product
    
    Args:
        product_description (str): Description of the body wash product
        
    Returns:
        str: Comma-separated tags
    """
    prompt = f"""### Instruction:
Analyze the following body wash product and predict its Level I factors (tags).

### Input:
Product: {product_description}

### Response:
Tags:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract tags
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("Tags:")[-1].strip()
    
    return response

# Example usage
product = "Lavender and chamomile body wash with moisturizing aloe vera"
tags = predict_bodywash_tags(product)
print(f"Predicted tags: {tags}")
```

### Batch Inference

```python
def predict_batch_bodywash_tags(product_descriptions, batch_size=4):
    """Predict tags for multiple products efficiently"""
    all_predictions = []
    
    for i in range(0, len(product_descriptions), batch_size):
        batch_descriptions = product_descriptions[i:i+batch_size]
        batch_predictions = []
        
        for desc in batch_descriptions:
            prompt = f"""### Instruction:
Analyze the following body wash product and predict its Level I factors (tags).

### Input:
Product: {desc}

### Response:
Tags:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("Tags:")[-1].strip()
            batch_predictions.append(response)
        
        all_predictions.extend(batch_predictions)
    
    return all_predictions

# Batch usage example
products = [
    "Refreshing mint body wash with tea tree oil",
    "Moisturizing shea butter body wash for dry skin",
    "Exfoliating body wash with sea salt and coconut"
]

predictions = predict_batch_bodywash_tags(products)
for product, tags in zip(products, predictions):
    print(f"Product: {product}")
    print(f"Tags: {tags}\n")
```

## Training Details

### Hyperparameters

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
)
```

### QLoRA Configuration

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Applications

- Product categorization and tagging systems
- E-commerce product classification
- Retail inventory management
- Market analysis and trend identification

## Limitations

- Trained specifically on body wash products
- Performance may vary with unconventional product descriptions
- Limited to the tag categories present in training data

## Citation

If you use this model in your research, please cite:

```bibtex
@software{smolLM_bodywash_tagging,
  title = {SmolLM-1.7B Bodywash Tagging Model},
  author = {Your Name},
  year = {2024},
  url = {https://huggingface.co/your-username/your-model-name}
}
```

