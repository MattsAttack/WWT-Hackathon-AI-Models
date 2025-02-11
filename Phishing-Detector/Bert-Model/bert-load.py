import pathlib
from typing import TYPE_CHECKING, Any, override

import pandas as pd
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TaskID
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (
    BatchEncoding,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import has_length

if TYPE_CHECKING:
    from rich.status import Status

p = pathlib.Path(__file__).parent.resolve().parent

# Load dataset
df = pd.read_csv(
    p / "CEAS_08.csv",
    encoding="latin1",
)

# Combine subject and body
df["text"] = f"{df['subject'].fillna('')} {df['body'].fillna('')}"

# Convert labels: 1 = Phishing, 0 = Legitimate
df["label"] = df["label"].astype(int)

# Split dataset into train/test
train_texts: list[str]
test_texts: list[str]
train_labels: list[int]
test_labels: list[int]
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
)


# Load tokenizer for DistilBERT
tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased",
)

# Tokenize text
train_encodings: BatchEncoding = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=512,
)
test_encodings: BatchEncoding = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=512,
)


class EmailDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, encodings: BatchEncoding, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


class RichProgressCallback(TrainerCallback):
    """A `TrainerCallback` that displays the progress of training or evaluation using Rich.

    From TRL, under the Apache License 2.0.
    """

    def __init__(self) -> None:
        self.training_bar: Progress | None = None
        self.prediction_bar: Progress | None = None

        self.training_task_id: TaskID | None = None
        self.prediction_task_id: TaskID | None = None

        self.rich_group: Live | None = None
        self.rich_console: Console | None = None

        self.training_status: Status | None = None
        self.current_step: int | None = None

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            self.training_bar = Progress()
            self.prediction_bar = Progress()

            self.rich_console = Console()

            self.training_status = self.rich_console.status("Nothing to log yet ...")

            self.rich_group = Live(
                Panel(
                    Group(self.training_bar, self.prediction_bar, self.training_status),
                ),
            )
            self.rich_group.start()

            self.training_task_id = self.training_bar.add_task(
                "[blue]Training the model",
                total=state.max_steps,
            )
            self.current_step = 0

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            assert self.training_bar is not None
            assert self.training_task_id is not None
            assert self.current_step is not None

            self.training_bar.update(
                self.training_task_id,
                advance=state.global_step - self.current_step,
                update=True,
            )
            self.current_step = state.global_step

    @override
    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        eval_dataloader: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero and has_length(eval_dataloader):
            assert eval_dataloader is not None
            assert self.prediction_bar is not None

            if self.prediction_task_id is None:
                self.prediction_task_id = self.prediction_bar.add_task(
                    "[blue]Predicting on the evaluation dataset",
                    total=len(eval_dataloader),
                )
            self.prediction_bar.update(self.prediction_task_id, advance=1, update=True)

    @override
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero and self.prediction_task_id is not None:
            assert self.prediction_bar is not None

            self.prediction_bar.remove_task(self.prediction_task_id)
            self.prediction_task_id = None

    @override
    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero and self.prediction_task_id is not None:
            assert self.prediction_bar is not None

            self.prediction_bar.remove_task(self.prediction_task_id)
            self.prediction_task_id = None

    @override
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        logs: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero and self.training_bar is not None:
            assert logs is not None
            assert self.training_status is not None

            _ = logs.pop("total_flos", None)
            self.training_status.update(f"[bold green]Status = {logs!s}")

    @override
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            assert self.rich_group is not None
            self.rich_group.stop()

            self.training_bar = None
            self.prediction_bar = None
            self.training_task_id = None
            self.prediction_task_id = None
            self.rich_group = None
            self.rich_console = None
            self.training_status = None
            self.current_step = None


trainer.add_callback(RichProgressCallback)
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
