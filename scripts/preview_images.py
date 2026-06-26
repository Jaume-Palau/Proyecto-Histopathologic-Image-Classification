

from src.dataset import DatasetWrapper_Train
from src.helper_functions import preview_images


training_set_wrapped = DatasetWrapper_Train()

print(f"Dataset size: {len(training_set_wrapped)}")

preview_images(
    dataset=training_set_wrapped,
    n_row=5,
    n_col=10
)