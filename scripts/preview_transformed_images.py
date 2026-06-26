from src.dataset import DatasetWrapper_Train
from src.helper_functions import preview_transformed_images

training_set_iterator = iter(DatasetWrapper_Train())

preview_transformed_images(
    training_set_iterator,
    n_row=5,
    n_col=10
)