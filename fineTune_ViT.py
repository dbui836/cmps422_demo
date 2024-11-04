from datasets import load_dataset
import numpy as np
import evaluate
import random
from PIL import ImageDraw, ImageFont, Image
from transformers import ViTImageProcessor
import torch
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer


# NOTE: This code file is a reproduction from a tutorial found on
# https://huggingface.co/blog/fine-tune-vit 
# The only modification made was that the original used a beans dataset
# whereas the code below has been slightly modified to use the cats vs dogs dataset

# GOAL: To fine tune a pretrained model on the kaggle cats vs dogs dataset

# Set data and pretrained model/processor
dataSet = load_dataset("cats_vs_dogs")
model = 'google/vit-base-patch16-224-in21k' # pretrained model trained on the ImageNet-21k, a vast dataset with 224x224 resolution images
processor = ViTImageProcessor.from_pretrained(model) # responsible for preprocessing the image (NOT breaking image into patches)

# transform processes each into into pixel and label values
def transform(example_batch):
    # For cats vs dog dataset, make sure to convert to RGB to avoid error
    inputs = processor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt') # last argument indicates return a tensor
    inputs['labels'] = example_batch['labels']
    return inputs

prepared_dataSet = dataSet.with_transform(transform)

# Collate functions organizes multiple pieces of data from samples in a batch together
def collate_fn(batch):
    
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]), # stack the tensors
        'labels': torch.tensor([x['labels'] for x in batch]) # return a 1D tensor of labels
    }

# Set metric to prioritize 
metric = evaluate.load("f1")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# Initialize model object
labels = dataSet['train'].features['labels'].names
model = ViTForImageClassification.from_pretrained(
    model,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# Configure 
training_args = TrainingArguments(
  output_dir="./vit-base-cats-vs-dogs-test",
  per_device_train_batch_size=32, # batches are division of training data
  eval_strategy="steps",
  num_train_epochs=4, # epochs = how many times go through entire data set
  fp16=True, # means try to use CUDA and GPU if available
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

# Create a training and validation split
train_test_split = prepared_dataSet['train'].train_test_split(test_size=0.2)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"],
    tokenizer=processor,
)

# Save weights and print results
train_results = trainer.train()
trainer.save_model("C:/Users/Danny/vit_fine_tuned-test")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()