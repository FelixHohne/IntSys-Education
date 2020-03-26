import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_loader import get_data_loaders
import typing

class LogisticRegressionModel(nn.Module):
    """LogisticRegressionModel is the logistic regression classifier.

    This class handles only the binary classification task.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    """
    def __init__(self, num_param):
        ## TODO 1: Set up network
        super(LogisticRegressionModel, self).__init__()
        self.num_param = num_param
        self.linear = nn.Linear(self.num_param, 1)

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = LogisticRegressionModel(1, logistic_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """

        ## TODO 2: Implement the logistic regression on sample x
        return F.sigmoid(self.linear(x.float()))


class MultinomialRegressionModel(nn.Module):
    """MultinomialRegressionModel is logistic regression for multiclass prob.

    This model operates under a one-vs-rest (OvR) scheme for its predictions.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    :param loss_fn: The loss function that is used to calculate "cost"
    :type loss_fn: typing.Callable[[torch.Tensor, torch.Tensor],torch.Tensor]

    .. seealso:: :class:`LogisticRegressionModel`
    """
    def __init__(self, num_param, loss_fn):
        ## TODO 3: Set up network
        # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
        pass

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = MultinomialRegressionModel(1, cross_entropy_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        ## TODO 4: Implement the logistic regression on sample x
        # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
        pass


def logistic_loss(output, target):
    """Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\dots,l_N\\}^\\top), \\quad
        l_n = -y_n \\cdot \\log x_n - (1 - y_n) \\cdot \\log (1 - x_n),

    where :math:`N` is the batch size.

    Note that the targets :math:`target` should be numbers between 0 and 1.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    # TODO 2: Implement the logistic loss function from the slides using
    # pytorch operations
    loss = -target*(torch.log(output))-(1-target)*(torch.log(output))
    return loss.mean()


def cross_entropy_loss(output, target):
    """Creates a criterion that measures the Cross Entropy
    between the target and the output:
    
    It is useful when training a classification problem with `C` classes.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
    return 0


if __name__ == "__main__":
    # TODO: Run a sample here
    # Look at linear_regression.py for a hint on how you should do this!!
    path_to_csv = 'data/DS3.csv'
    train_val_test = [0.6, 0.2, 0.2]
    batch_size = 32
    num_param = 2
    lr = 0.0001
    loss_fn = logistic_loss
    TOTAL_TIME_STEPS = 1000

    train_loader, val_loader, test_loader =\
       get_data_loaders(path_to_csv,
                        train_val_test=train_val_test,
                        batch_size=batch_size)

    model = LogisticRegressionModel(num_param)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for t in range(TOTAL_TIME_STEPS):
       for batch_index, (input_t, y) in enumerate(train_loader):

            optimizer.zero_grad()
            preds = model(input_t)
            #print(preds)
            #print(y)
            loss = loss_fn(preds, y.view(1,len(y)))  # You might have to change the shape of things here.
            loss.backward()
            #print(loss)
            optimizer.step()
    
    model.eval()
    for batch_index, (input_t, y) in enumerate(val_loader):
    
        preds = model(input_t)
    
        loss = loss_fn(preds, y.view(1,len(y)))
        #print(loss)