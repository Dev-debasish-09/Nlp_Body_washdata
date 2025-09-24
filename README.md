---
base_model: HuggingFaceTB/SmolLM-1.7B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:HuggingFaceTB/SmolLM-1.7B-Instruct
- lora
- transformers
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.17.1



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

## Contact

For questions or issues regarding this model, please open an issue on the model repository.
