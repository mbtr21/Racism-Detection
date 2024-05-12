from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch


class RegenerateDataset:
    def __init__(self, left_dataset, right_dataset):
        self.data_frame = None
        self.left_dataset = pd.read_csv(f'{left_dataset}')
        self.right_dataset = pd.read_csv(f'{right_dataset}')
        self.tokenizer = BertTokenizer.from_pretrained('bert_finetuned')
        self.model = BertForSequenceClassification.from_pretrained('bert_finetuned')

    def generate_dataset(self):
        import pandas as pd
        final_data = dict()
        final_data['text'] = list(self.left_dataset['text'])
        final_data['label'] = list(self.left_dataset['label'])
        final_data['text'] += list(self.right_dataset['text'])
        final_data['label'] += list(self.right_dataset['label'])
        self.data_frame = pd.DataFrame(final_data)

    def predict_dataset(self):
        self.data_frame['label'] = self.data_frame['label'].replace(1, 3)

        # Iterate through DataFrame and update labels where current label is 0
        for index in range(len(self.data_frame)):
            if self.data_frame.at[index, 'label'] == 0:  # Use .at for more efficient and correct assignment
                inputs = self.tokenizer(self.data_frame.at[index, 'text'], return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_index = probabilities.argmax().item()

                # Set the predicted class index safely using .at
                self.data_frame.at[index, 'label'] = predicted_class_index
                print(self.data_frame.at[index, 'label'])

        # Save the updated DataFrame to a CSV file
        self.data_frame.to_csv('updated_dataset.csv')


regenerate_dataset = RegenerateDataset('RacismDetectionDataset.csv', 'twitter_racism.csv')
regenerate_dataset.generate_dataset()
regenerate_dataset.predict_dataset()
