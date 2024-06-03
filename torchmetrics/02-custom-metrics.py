"""
author: rohan singh
python module to implement some custom metrics

implementing your own metric is as easy as subclassing a torch.nn.Module. simply, subclass Metric and do the following:
    - implement __init__ where you call self.add_state for every internal state that is needed for the metrics computations
    - implement update method, where all logic that is necessary for updating metric states go
    - implement compute method, where the final metric computations happens
"""


# imports
import torch
from torchmetrics import Metric
from torch import Tensor



# subclassing to define the custom metric
class FooMetric(Metric):

    # defining the states for the metric computations
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    # logic for updating the metrics
    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()


    # actual metric computation
    def compute(self) -> Tensor:
        return self.correct.float() / self.total