from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import  Dataset
import pandas as pd
import torch


class BertTrainer:
    def __init__(self, dataset, model_path):
        """
        Initialize the BertTrainer class.

        Args:
            dataset (str): File path of the dataset.
            model_path (str): Path to the pre-trained BERT model.
        """
        self.dataset = pd.read_csv(f'{dataset}')
        self.tokenizer = BertTokenizer.from_pretrained(f'{model_path}')
        num_labels = self.dataset['label'].nunique()
        self.model = BertForSequenceClassification.from_pretrained(f'{model_path}', num_labels=num_labels)
        self.dataset.dropna(subset=['tweet', 'label'], inplace=True)
        self.dataset['tweet'] = self.dataset['tweet'].astype(str)
        self.dataset['label'] = self.dataset['label'].astype(int)

    def tokenize_function(self, examples):
        """
        Tokenization function to tokenize text samples.

        Args:
            examples (dict): Dictionary containing text samples.

        Returns:
            dict: Tokenized text samples.
        """
        return self.tokenizer(examples['tweet'], padding="max_length", truncation=True, max_length=128)

    def data_process(self):
        """
        Process the dataset by tokenizing text samples and splitting into train and test sets.

        Returns:
            Dataset, Dataset: Train and test datasets.
        """
        hf_dataset = Dataset.from_pandas(self.dataset)
        tokenized_datasets = hf_dataset.map(self.tokenize_function, batched=True)
        train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=0.1).values()
        return train_dataset, test_dataset

    def train(self):
        """
        Train the BERT model using the specified training arguments.
        """
        train_dataset, test_dataset = self.data_process()
        training_args = TrainingArguments(
            output_dir='./results',            # Where to store the model outputs
            num_train_epochs=3,                # Number of training epochs
            per_device_train_batch_size=16,    # Batch size for training
            per_device_eval_batch_size=16,     # Batch size for evaluation
            weight_decay=0.01,                 # Regularization to prevent overfitting
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="no",          # Skips evaluation to speed up training
            save_strategy="no",                # Skips saving the model to speed up training
            disable_tqdm=True                  # Disables tqdm progress bars to reduce overhead
        )
        # Check for CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)  # Move model to the appropriate device

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        trainer.train()
        trainer.evaluate()
        model_path = './bert_finetuned'
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)


bert_trainer = BertTrainer('train.csv','bert-base-uncased')
bert_trainer.train()
