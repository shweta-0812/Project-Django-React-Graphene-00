from typing import Any

import torch
from datasets import load_dataset, metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from tqdm.auto import tqdm
import collections

from transformers import TrainingArguments
from transformers import Trainer


def compute_metrics_for_model(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.Metric.compute(predictions=predicted_answers, references=theoretical_answers)


class LmTransformerTrainer:

    def __init__(self):
        self.preprocess_validation_examples = None
        self.validation_dataset = None
        self.train_dataset = None
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(device)

        self.raw_datasets = load_dataset("squad")

        # During training, there should be only one possible answer, so checking the number of answers in the
        # training set
        print(self.raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1))

        # During validation, there are more than one possible answer to validate against predictions, so checking the
        # number of answers in the validation set
        print(self.raw_datasets["validation"].filter(lambda x: len(x["answers"]["text"]) != 1))

        self.model_checkpoint = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint)
        self.max_length = 384
        self.stride = 128

    def preprocess_training_datasets(self, train_ds):
        # string clean
        questions: list[Any] = [q.strip() for q in train_ds["question"]]
        answers = train_ds["answers"]

        inputs = self.tokenizer(
            questions,
            train_ds["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # offset_mapping for input ids to context
        offset_map_list = inputs.pop("offset_mapping")
        # overflow fragment context mapping to original sample/data/context
        sample_map_list = inputs.pop("overflow_to_sample_mapping")

        ans_start_positions = []
        ans_end_positions = []

        for i, offset_map in enumerate(offset_map_list):

            sample_idx = sample_map_list[i]
            answer = answers[sample_idx]

            ans_start_idx = answer["answer_start"][0]
            ans_end_idx = answer["answer_start"][0] + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            j = 0
            while sequence_ids[j] != 1:
                j += 1
            context_start = j

            while sequence_ids[j] == 1:
                j += 1
            context_end = j - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset_map[context_start][0] > ans_start_idx or offset_map[context_end][1] < ans_end_idx:
                ans_start_positions.append(0)
                ans_end_positions.append(0)
            else:
                # otherwise it's the start and end token positions
                k = context_start
                while k <= context_end and offset_map[k][0] <= ans_start_idx:
                    k += 1
                ans_start_positions.append(k - 1)

                k = context_end
                while k >= context_start and offset_map[k][1] >= ans_end_idx:
                    k -= 1
                ans_end_positions.append(k + 1)

        inputs["start_positions"] = ans_start_positions
        inputs["end_positions"] = ans_end_positions
        return inputs

    def prep_training_datasets(self):
        self.train_dataset = self.raw_datasets["train"].map(
            self.preprocess_training_datasets,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
        )
        len(self.raw_datasets["train"]), len(self.train_dataset)

    def preprocess_validation_datasets(self, valid_ds):
        questions = [q.strip() for q in valid_ds["question"]]
        dataset_ids = []

        inputs = self.tokenizer(
            questions,
            valid_ds["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map_list = inputs.pop("overflow_to_sample_mapping")

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map_list[i]
            dataset_ids.append(valid_ds["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["dataset_id"] = dataset_ids
        return inputs

    def prep_validation_datasets(self):
        self.validation_dataset = self.raw_datasets["validation"].map(
            self.preprocess_validation_examples,
            batched=True,
            remove_columns=self.raw_datasets["validation"].column_names,
        )
        len(self.raw_datasets["validation"]), len(self.validation_dataset)

    def train_compute_metrci_and_save_model(self):
        args = TrainingArguments(
            output_dir="fine_tuned_lms",
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=False,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()

        predictions, _, _ = trainer.predict(self.validation_dataset)
        start_logits, end_logits = predictions
        model_metric = compute_metrics_for_model(start_logits, end_logits, self.validation_dataset, self.raw_datasets["validation"])
        print(model_metric)


if __name__ == "__main__":
    print("::::step 1:::::::::::")
    lm_trainer = LmTransformerTrainer()
    print("::::step 2:::::::::::")
    lm_trainer.prep_training_datasets()
    print("::::step 3:::::::::::")
    lm_trainer.prep_validation_datasets()
    print("::::step 4:::::::::::")
    lm_trainer.train_compute_metrci_and_save_model()

