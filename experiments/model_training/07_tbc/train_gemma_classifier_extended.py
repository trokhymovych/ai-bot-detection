import glob
import json
import os
import joblib
from tqdm import tqdm
import pandas as pd

from typing import Dict

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from datasets import ClassLabel, Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file

MODELS_PATH = "../../../models/"
RANDOM_SEED = 42

gemma_params = {
    "initial_model": "google/gemma-3-1b-pt",
    "batch_size": 4,
    "model_name": "gemma3_1b_extended",
    "max_length": 512,
    "n_epochs": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "gradient_accumulation_steps": 16,  # effective batch ~64
}


def _restore_cls_head(peft_model: PeftModel, checkpoint: str) -> None:
    ckpt_files = glob.glob(os.path.join(checkpoint, "*.safetensors"))
    if not ckpt_files:
        return
    state_dict = load_file(ckpt_files[0])
    if "base_model.model.score.weight" not in state_dict:
        return
    target = peft_model.base_model.model.score.modules_to_save["default"].weight
    target.data.copy_(state_dict["base_model.model.score.weight"].to(target.dtype))
    print("Classification head restored from checkpoint.")


class TRClassifier:
    """LoRA-based sequence classifier built on a decoder LLM."""

    name = "Transformer based classifier"
    allows_classification: bool = True

    def __init__(self, params: Dict = {}):
        self.model_checkpoint = params.get("initial_model")
        self.batch_size = params.get("batch_size")
        self.model_name = params.get("model_name", "gemma")
        self.max_length = params.get("max_length")
        self.n_epochs = params.get("n_epochs")
        self.gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)

        is_merged = (
            os.path.exists(os.path.join(self.model_checkpoint, "config.json")) and
            os.path.exists(os.path.join(self.model_checkpoint, "tokenizer_config.json"))
        )

        if is_merged:
            # Inference path: LoRA already baked into weights, load as standard model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=2, torch_dtype=torch.bfloat16,
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.eval()
        else:
            # Training path: load base model and wrap with LoRA
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=2, torch_dtype=torch.bfloat16,
            )
            base_model.config.pad_token_id = self.tokenizer.pad_token_id
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=params.get("lora_r", 16),
                lora_alpha=params.get("lora_alpha", 32),
                lora_dropout=params.get("lora_dropout", 0.1),
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()

    def fit(self, trainset: pd.DataFrame, **kwargs) -> None:
        encoded_dataset = self._prepare_data(trainset)
        args = TrainingArguments(
            f"{MODELS_PATH}{self.model_name}_tuned/",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.n_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            bf16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
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
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.model = trainer.model

        if trainer.state.best_model_checkpoint:
            _restore_cls_head(self.model, trainer.state.best_model_checkpoint)

        merged_dir = f"{MODELS_PATH}{self.model_name}_merged/"
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)
        print(f"\nMerged model saved to: {merged_dir}")
        print(f"Use this path for inference: {merged_dir}")

    def predict(self, testset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if torch.cuda.is_available():
            device = 0
        else:
            device = "cpu"

        classifier = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            batch_size=self.batch_size,
        )

        sentences = []
        for _, row in testset.iterrows():
            sentences.append(row['real_text'])
            sentences.append(row['generated_text'])

        sentences = list(set(sentences))

        texts_length = [len(t) for t in sentences]
        ids = np.argsort(texts_length)[::-1]
        sentences_sorted = [sentences[i] for i in ids]

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
                batch_size=self.batch_size,
            )
        predicted_scores = predictions_processing(predicted_scores)

        text_to_score = {text: score for text, score in zip(sentences_sorted, predicted_scores)}
        testset[f"tc_{self.model_name}_real_score"] = testset['real_text'].map(text_to_score)
        testset[f"tc_{self.model_name}_generated_score"] = testset['generated_text'].map(text_to_score)
        return testset

    def _prepare_data(self, trainset: pd.DataFrame):
        train, validation = [], []
        for _, row in trainset.iterrows():
            has_real = pd.notna(row['real_text']) and row['real_text']
            has_gen = pd.notna(row['generated_text']) and row['generated_text']
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
            tokenized = self.tokenizer(examples["text"], truncation=True, max_length=self.max_length)
            # Gemma 3 requires token_type_ids for its mixed local/global attention mask
            tokenized["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]
            return tokenized

        encoded_dataset = training_dataset.map(preprocess_function, batched=True)
        return encoded_dataset

    @classmethod
    def convert_checkpoint(cls, checkpoint: str, merged_dir: str) -> None:
        """Convert an existing Trainer LoRA checkpoint to a merged model without retraining."""
        with open(os.path.join(checkpoint, "adapter_config.json")) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=2, torch_dtype=torch.bfloat16,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id

        peft_model = PeftModel.from_pretrained(base_model, checkpoint)
        _restore_cls_head(peft_model, checkpoint)

        merged = peft_model.merge_and_unload()
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


if __name__ == "__main__":
    train = pd.read_parquet("../../../data/train_data_extended.parquet")
    print("Number of samples to process train:", len(train))

    tc_gemma = TRClassifier(params=gemma_params)
    tc_gemma.fit(train)
