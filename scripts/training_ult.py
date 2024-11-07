"""Training with ULT for Text."""

from ult.data.text.core import WikiText103Loader
from ult.configuration import UnicodeTextBWRLETransformParameters
from ult.transforms import UnicodeTextBWRLETransform
from loguru import logger
from datasets import DatasetDict, IterableDatasetDict
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Constants, modify as needed
FIELD_NAME = "text"
NUM_WORKERS = 8
CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_PATH = Path("ult_trainer")
PATH_TO_LUT = OUTPUT_PATH.joinpath("LUT.json")

# Load Dataset
dataset = WikiText103Loader()
logger.info("Data loaded")

# Inizialize ULT transformation to Unicode strings
unicode_configuration = UnicodeTextBWRLETransformParameters(patch_size=10)
unicoder = UnicodeTextBWRLETransform(unicode_configuration)

# Compute LUT for the training data
logger.info("Data preprocessing...")
if isinstance(dataset, DatasetDict):
    instances = list(dataset["train"][FIELD_NAME])
    unicoder.add_multiple_instances(instances=instances, num_workers=NUM_WORKERS)
# NOTE: If the dataset is big list(dataset["train"][FIELD_NAME]) explodes OOM,
# betther to use lazy IterableDatasetDict
elif isinstance(dataset, IterableDatasetDict):
    batched_dataset = dataset["train"].batch(batch_size=32)
    for sample in iter(batched_dataset):
        instances = sample[FIELD_NAME]
        unicoder.add_multiple_instances(instances, num_workers=NUM_WORKERS)
else:
    raise ValueError(
        f"Dataset has to be of type DatasetDict or IterableDatasetDict but received {type(dataset)}."
    )
unicoder.save_occurrences(PATH_TO_LUT)

# Apply preprocessing to the dataset
preprocessed_dataset = dataset.map(function=unicoder.encode, input_columns=FIELD_NAME)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenized_dataset = preprocessed_dataset.map(
    function=tokenizer, input_columns=FIELD_NAME, fn_kwargs={"truncation": True}
)

# Model
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)

# Training
training_args = TrainingArguments(
    output_dir=str(OUTPUT_PATH), evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
)

trainer.train()
