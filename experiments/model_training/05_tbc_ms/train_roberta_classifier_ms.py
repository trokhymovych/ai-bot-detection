import joblib
from tqdm import tqdm
import pandas as pd

from typing import Dict
from collections import defaultdict

import numpy as np
import torch
import random

import evaluate
from datasets import ClassLabel, Dataset, DatasetDict
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

BASE_MODEL = "FacebookAI/xlm-roberta-base"
MODEL_NAME = "xlmroberta_ms"
MODELS_PATH = "../../../models/"
MAX_LENGTH = 512
N_EPOCHS = 4
RANDOM_SEED = 42
BATCH_SIZE = 64

bert_params = {
    "initial_model": BASE_MODEL, 
    "batch_size": BATCH_SIZE,
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "n_epochs": N_EPOCHS
}


class TRClassifier:
    """
    Transformer based classifier model
    """

    name = "Transformer based classifier"
    allows_classification: bool = True

    def __init__(self, params: Dict = {}):
        # model initialization:
        self.model_checkpoint = params.get("initial_model", BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=2)
        self.batch_size = params.get("batch_size", BATCH_SIZE)
        self.classifier = None
        self.model_name = params.get("model_name", "bert")
        self.max_length = params.get("max_length", MAX_LENGTH)
        self.n_epochs = params.get("n_epochs", N_EPOCHS)

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
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        metric = evaluate.load("glue", "mrpc")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

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
                res.append(i[1]['score']) # score for "generated" label
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

        size = len(trainset)
        true_ratio = 0.05
        seed = RANDOM_SEED
        rng = np.random.default_rng(seed)
        n_true = int(size * true_ratio)
        vec = np.array([True] * n_true + [False] * (size - n_true))
        rng.shuffle(vec)
        trainset['is_val'] = vec
        
        for _, row in trainset.iterrows():
            if row['is_val']:
                validation.append({"text": row['text'], "label": "real" if row["label"] == 0 else "generated"})
            else:
                train.append({"text": row['text'], "label": "real" if row["label"] == 0 else "generated"})

        print("Training size: ", len(train))
        # --- Downsampling ---
        by_class = defaultdict(list)
        for ex in train:
            by_class[ex["label"]].append(ex)
        
        # Identify minority size
        min_class_size = min(len(v) for v in by_class.values())
        # Downsample all classes to minority size (majority-only if 2 classes)
        downsampled_train = []
        for label, examples in by_class.items():
            if len(examples) > min_class_size:
                rng_random = random.Random(RANDOM_SEED)
                examples = rng_random.sample(examples, min_class_size)
            downsampled_train.extend(examples)
        
        # Deterministic shuffle
        rng.shuffle(downsampled_train)

        training_dataset = DatasetDict({
            "train": Dataset.from_list(downsampled_train),
            "test": Dataset.from_list(validation),
        })
        feat_class = ClassLabel(num_classes=2, names=["real", "generated"])
        print("Final Training size: ", len(training_dataset["train"]))
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

    multisocial = pd.read_csv("../../../data/external_data/multisocial/multisocial_anonymized.csv")
    train = multisocial[multisocial.split == "train"]
    test = multisocial[multisocial.split == "test"]
    print("Number of samples to process train:", len(train))
    print("Number of samples to process test:", len(test))
