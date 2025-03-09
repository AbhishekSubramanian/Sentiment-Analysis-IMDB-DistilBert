# IMDB-Sentiment-Classifier-DistilBert

# IMDB Movie Reviews Sentiment Analysis

This project demonstrates sentiment analysis on the IMDB Movie Reviews dataset using the DistilBERT model from Hugging Face. The goal is to classify reviews as either positive or negative sentiments.

## Dataset

- **Dataset Name**: IMDB Movie Reviews
- **Source**: [Hugging Face Datasets - IMDB](https://huggingface.co/datasets/imdb)
- **Total Reviews**: 50,000 (25,000 training, 25,000 testing)
- **Labels**: Positive (1), Negative (0)

## Model Overview

- **Model**: DistilBERT (a smaller, faster variant of BERT)
- **Architecture**: Transformer-based
- **Purpose**: Text classification (sequence classification)
- **Library**: [Transformers](https://huggingface.co/transformers)

## Installation

To install necessary dependencies, run:

```bash
pip install transformers datasets evaluate torch
```

## Running the Project

### 1. Load and Explore Dataset

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

### 2. Tokenize the Data

```python
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
```

### 3. Train the Model

```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'].shuffle(seed=42).select(range(5000)),
    eval_dataset=tokenized_dataset['test'].shuffle(seed=42).select(range(1000)),
)

trainer.train()
```

### 4. Making Predictions

```python
import torch

model_path = "./results/checkpoint-626"  # Replace with your latest checkpoint
model = DistilBertForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    prediction = probabilities.argmax().item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[0][prediction].item()
    return sentiment, confidence

# Example usage
review = "The movie was fantastic!"
sentiment, confidence = predict_sentiment(review)
print(f"Sentiment: {sentiment}, Confidence: {confidence*100:.2f}%")
```

## Viewing Training Metrics

To visualize training metrics using TensorBoard, run:

```python
%load_ext tensorboard
%tensorboard --logdir ./results/runs/
```

## Results

The fine-tuned DistilBERT model achieved approximately 90% accuracy on the evaluation dataset, effectively distinguishing between positive and negative sentiments in IMDB movie reviews.

## Useful Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

