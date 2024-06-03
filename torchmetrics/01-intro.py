"""
author: rohan singh
simple introduction to built in performance metrics of the torchmetrics library
"""


# imports
import torch
import torchmetrics


# main function
def main():
    
    # initializing the metric
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)


    # random classifier
    n_batches = 10
    for i in range(n_batches):
   
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
    
        acc = metric(preds, target)
        print(f"accuracy on batch {i}: {acc}")

    acc = metric.compute()
    print(f"accuracy on all data: {acc}")


    # note it is important to reset the metric after use
    metric.reset()




if __name__ == "__main__":
    main()

