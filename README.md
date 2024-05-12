# Racism Detection Project

This project focuses on the automatic detection of racist text using machine learning models. The approach combines dataset preparation, model training, and evaluation.

## Project Structure

* **`trainer_for_dataset.py`:**  The core script for training a BERT classifier to predict the class (racist/non-racist) of text data. This should be run first.
* **`regenrate_dataset.py`:** This script is used to create or preprocess the dataset required for training and evaluation.
* **`Bert.ipynb`:** A Jupyter Notebook containing the implementation of an ALBERT model for racism detection.
* **`lstm.ipynb`:** A Jupyter Notebook containing the implementation of an LSTM model for racism detection.

## Getting Started

1. **Dataset Preparation:**
   * Run `python regenrate_dataset.py` to generate or preprocess your dataset. Ensure the output format is suitable for the BERT classifier.

2. **Model Training:**
   * Execute `python trainer_for_dataset.py`. This will train the BERT classifier on your prepared dataset. The script may require you to specify paths to your data files and model output directory.
   
3. **Model Evaluation (ALBERT and LSTM):**
   * Open and run the `Bert.ipynb` and `lstm.ipynb` notebooks. These notebooks contain code to load your trained models, evaluate them on test data, and potentially fine-tune them for better performance.

## Dependencies

* Python 3.x
* Transformers library (Hugging Face)
* PyTorch
* Jupyter Notebook
* Other required libraries (install via `pip install -r requirements.txt`)

## Customization

* Feel free to modify the hyperparameters in `trainer_for_dataset.py` to experiment with different settings for the BERT model.
* Adapt the dataset preparation (`regenrate_dataset.py`) to your specific data source and format.
* Explore the ALBERT and LSTM models in their respective notebooks, fine-tune them, or even add new models to the project.

## Additional Notes

* Ensure you have a suitable dataset for training and evaluation. Consider ethical implications and potential biases when working with sensitive data.
* Regularly update the models with new data to maintain their performance and relevance.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests to improve this project.


