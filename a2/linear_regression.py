import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import data_loader as dl
import plotting as plot

class LinearRegressionModel(nn.Module):
    """LinearRegressionModel is the linear regression regressor.

    This class handles only the standard linear regression task.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    """
    def __init__(self, num_param):
        ## TODO 1: Set up network
        super(LinearRegressionModel, self).__init__()
        self.num_param = num_param
        self.thetas = torch.nn.Parameter(torch.randn(1, self.num_param))

    def forward(self, x):
        """forward generates the predictions for the input

        This function does not have to be called explicitly. We can do the
        following

        .. highlight:: python
        .. code-block:: python

            model = LinearRegressionModel(1, mse_loss)
            predictions = model(X)

        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        ## TODO 2: Implement the linear regression on sample x
        return torch.mm(self.thetas,torch.t(x).float())


def data_transform(sample):

  ## TODO: Define a transform on a given (x, y) sample. This can be used, for example
  ## for changing the feature representation of your data so that Linear regression works
  ## better.
  return np.append([1],sample)


def mse_loss(output, target):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`output` and target :math:`target`.

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left( x_n - y_n \\right)^2,

    where :math:`N` is the batch size.

    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 3: Implement Mean-Squared Error loss.
    # Use PyTorch operations to return a PyTorch tensor
    assert target.shape == output.shape
    loss = output.sub(target)
    loss = torch.pow(loss, 2)
    assert loss.shape == output.shape
    mse = torch.sum(loss)
    size = target.shape
    n = size[1]
    #change to mean later -E
    return torch.div(mse,n)


def mae_loss(output, target):
    """Creates a criterion that measures the mean absolute error (l1 loss)
    between each element in the input :math:`output` and target :math:`target`.

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left| x_n - y_n \\right|,

    where :math:`N` is the batch size.

    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 4: Implement L1 loss. Use PyTorch operations.
    # Use PyTorch operations to return a PyTorch tensor.
    assert target.shape == output.shape
    loss = output.sub(target)
    loss = torch.abs(loss)
    assert loss.shape == output.shape
    mse = torch.sum(loss)
    size = target.shape
    n = size[1]
    #change to mean later -E
    return torch.mean(loss)

def percent_error(output, target):
    loss = torch.abs(torch.div(torch.abs(target-output), target))
    return torch.mean(loss)*100

if __name__ == "__main__":
    ## Here you will want to create the relevant dataloaders for the csv files for which
    ## you think you should use Linear Regression. The syntax for doing this is something like:
    # Eg:
    # train_loader, val_loader, test_loader =\
    #   get_data_loaders(path_to_csv,
    #                    transform_fn=data_transform  # Can also pass in None here
    #                    train_val_test=[YOUR TRAIN/VAL/TEST SPLIT],
    #                    batch_size=YOUR BATCH SIZE)


    ## Now you will want to initialise your Linear Regression model, using something like
    # Eg:
    # model = LinearRegressionModel(...)


    ## Then, you will want to define your optimizer (the thing that updates your model weights)
    # Eg:
    # optimizer = optim.[one of PyTorch's optimizers](model.parameters(), lr=0.01)


    ## Now, you can start your training loop:
    # Eg:
    # model.train()
    # for t in range(TOTAL_TIME_STEPS):
    #   for batch_index, (input_t, y) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #
    #     preds = Feed the input to the model
    #
    #     loss = loss_fn(preds, y)  # You might have to change the shape of things here.
    #
    #     loss.backward()
    #     optimizer.step()
    #
    ## Don't worry about loss.backward() for now. Think of it as calculating gradients.



    ## And voila, your model is trained. Now, use something similar to run your model on
    ## the validation and test data loaders:
    # Eg:
    # model.eval()
    # for batch_index, (input_t, y) in enumerate(val/test_loader):
    #
    #   preds = Feed the input to the model
    #
    #   loss = loss_fn(preds, y)
    #
    ## You don't need to do loss.backward() or optimizer.step() here since you are no
    ## longer training.

    path_to_csv = 'data/DS1.csv'
    ds1df= pd.read_csv('data/DS1.csv')
    dataset1=ds1df.to_numpy()
    transform_fn = data_transform  # Can also pass in None here
    train_val_test = [0.8, 0.1, 0.1]
    batch_size = 32
    num_param = 3
    lr = 0.01
    loss_fn = mae_loss
    TOTAL_TIME_STEPS = 100

    train_loader, val_loader, test_loader =\
        dl.get_data_loaders(
            path_to_csv,
            transform_fn=transform_fn,
            train_val_test=train_val_test,
            batch_size=batch_size)

    model = LinearRegressionModel(num_param)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for t in range(TOTAL_TIME_STEPS):
       for batch_index, (input_t, y) in enumerate(train_loader):

            optimizer.zero_grad()
            #print(model.thetas)
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

        print(preds)
        loss = loss_fn(preds, y.view(1,len(y)))

        """Uncomment below for the percent error across the eval set"""

        #percentError = percent_error(preds, y.view(1, len(y)))
        #print(percentError)

        """Function plot_linear is used for Dataset 1 because it has two features.
        Use function plot_linear_2D for Dataset 2."""

        plot.plot_linear(preds, input_t, y.view(1, len(y)))