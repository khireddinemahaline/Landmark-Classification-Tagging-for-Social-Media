import torch
import torch.nn as nn

# define the CNN architecture

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        
        super(MyModel, self).__init__()
        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1), # 16 x 224 x 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 112 x 112

            nn.Conv2d(16, 32, kernel_size = 3, padding = 1), # 32 x 112 x 112    
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 56 x 56

            nn.Conv2d(32, 32, kernel_size = 3, padding = 1), # 32 x 56 x 56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 28 x 28

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), # 64 x 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 14 x 14

            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 x 7 x 7
           
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), # 64 x 7 x 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten(),  # 64 x 7 x 7

            nn.Linear(64 * 7 * 7, 500),  # -> 500
            nn.Dropout(dropout),
            nn.BatchNorm1d(500),

            nn.ReLU(),
            nn.Linear(500, num_classes)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # YOUR CODE HERE: process the input tensor through the
         # feature extractor, the pooling and the final linear
         # layers (if appropriate for the architecture chosen)
       
        return self.model(x)
        

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"