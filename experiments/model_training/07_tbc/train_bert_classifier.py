import joblib
from tqdm import tqdm
import pandas as pd

from typing import Dict

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from datasets import ClassLabel, Dataset, DatasetDict
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

MODELS_PATH = "../../../models/"
RANDOM_SEED = 42

bert_params = {
    "initial_model": "bert-base-multilingual-cased", 
    "batch_size": 64,
    "model_name": "mbert",
    "max_length": 512,
    "n_epochs": 3
}

class TRClassifier:
    """
    Transformer based classifier model
    """

    name = "Transformer based classifier"
    allows_classification: bool = True

    def __init__(self, params: Dict = {}):
        # model initialization:
        self.model_checkpoint = params.get("initial_model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=2)
        self.batch_size = params.get("batch_size")
        self.classifier = None
        self.model_name = params.get("model_name", "bert")
        self.max_length = params.get("max_length")
        self.n_epochs = params.get("n_epochs")

    def fit(self, trainset: pd.DataFrame, **kwargs) -> None:
        """
        Method used for model training. Parameters are hardcoded.
        """
        encoded_dataset = self._prepare_data(trainset)
        args = TrainingArguments(
            f"{MODELS_PATH}{self.model_name}_tuned/",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.n_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            probs = softmax(logits, axis=1)[:, 1]
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions),
                "auc": roc_auc_score(labels, probs),
            }

        trainer = Trainer(
            self.model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        self.model = trainer.model

    def predict(self, testset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Method used for batch predict scores for the whole dataset
        """
        # Creating classification pipeline
        if torch.cuda.is_available():
            device = 0
        else:
            device = "cpu"

        classifier = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            batch_size=self.batch_size
        )

        # Preparing texts for prediction
        sentences = []
        for _, row in testset.iterrows():
            sentences.append(row['real_text'])
            sentences.append(row['generated_text'])
        
        # Deduplication of sentences
        sentences = list(set(sentences))

        # Sorting sentences by the length to make the batch processing more efficient:
        texts_length = [len(t) for t in sentences]
        ids = np.argsort(texts_length)[::-1]
        sentences_sorted = [sentences[i] for i in ids]

        # Predicting using pipeline:
        def predictions_processing(predictions):
            res = []
            for i in predictions:
                if i["label"] == "LABEL_0":
                    res.append(1 - i["score"])
                elif i["label"] == "LABEL_1":
                    res.append(i["score"]) 
                else: 
                    raise ValueError
            return res
        
        tqdm_batch_size = self.batch_size * 10
        predicted_scores = []
        for i in tqdm(range(0, len(sentences_sorted), tqdm_batch_size)):
            texts_batch = sentences_sorted[i:i + tqdm_batch_size]
            tokenizer_kwargs = {'truncation': True, 'max_length': self.max_length}
            predicted_scores += classifier(
                texts_batch,
                return_all_scores=True,
                **tokenizer_kwargs,
                batch_size=self.batch_size
            )
        predicted_scores = predictions_processing(predicted_scores)



        text_to_score = {text: score for text, score in zip(sentences_sorted, predicted_scores)}
        testset[f"tc_{self.model_name}_real_score"] = testset['real_text'].map(text_to_score)
        testset[f"tc_{self.model_name}_generated_score"] = testset['generated_text'].map(text_to_score)
        return testset

    def _prepare_data(self, trainset: pd.DataFrame):
        """
        Method that process the training dataset to use it for training
        :param trainset:
        :return:
        """
        train, validation = [], []
        for _, row in trainset.iterrows():
            has_real = pd.notna(row['real_text']) and row['real_text']
            has_gen = pd.notna(row['real_text']) and row['real_text']
            if row['is_val']:
                if has_real:
                    validation.append({"text": row['real_text'], "label": "real"})
                if has_gen:
                    validation.append({"text": row['generated_text'], "label": "generated"})
            else:
                if has_real:
                    train.append({"text": row['real_text'], "label": "real"})
                if has_gen:
                    train.append({"text": row['generated_text'], "label": "generated"})
        
        training_dataset = DatasetDict({
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(validation),
        })
        feat_class = ClassLabel(num_classes=2, names=["real", "generated"])
        print("Training size: ", len(training_dataset["train"]))
        print("Validation size: ", len(training_dataset["test"]))

        training_dataset = training_dataset.cast_column("label", feat_class)

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=self.max_length)

        encoded_dataset = training_dataset.map(preprocess_function, batched=True)
        return encoded_dataset
    
    def save(self, filename):
        # Use joblib to save the class method
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        # Use joblib to load the class method
        loaded_instance = joblib.load(filename)
        return loaded_instance

if __name__ == "__main__":
    train = pd.read_parquet("../../../data/train_data.parquet")
    print("Number of samples to process train:", len(train))
    test = pd.read_parquet("../../../data/test_data.parquet")
    print("Number of samples to process test:", len(test))
    
    tc_bert = TRClassifier(params=bert_params)
    tc_bert.fit(train)

