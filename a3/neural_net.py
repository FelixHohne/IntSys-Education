import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_loader import get_data_loaders
from typing import List, Union, Tuple

class SimpleNeuralNetModel(nn.Module):
    """SimpleNeuralNetModel [summary]
    
    [extended_summary]
    
    :param layer_sizes: Sizes of the input, hidden, and output layers of the NN
    :type layer_sizes: List[int]
    """
    def __init__(self, layer_sizes: List[int]):
        super(SimpleNeuralNetModel, self).__init__()
        # TODO: Set up Neural Network according the to layer sizes
        # The first number represents the input size and the output would be
        # the last number, with the numbers in between representing the
        # hidden layer sizes
        self.linears = nn.ModuleList()
        self.softmax = nn.Softmax(dim = 1)
        self.loss = nn.NLLLoss()
        length = len(layer_sizes)
        for i in range (0, length -1): 
            self.linears.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, x):
        """forward generates the prediction for the input x.
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        #TODO: Matrix mult then relu
        m,n = x.size
        for i in range (0,n-1):
            l = self.linears[i]
            x = nn.ReLU(x * l)
        x = self.softmax(x * self.linears[n])
        return np.to_array(x)


class SimpleConvNetModel(nn.Module):
    """SimpleConvNetModel [summary]
    
    [extended_summary]
    
    :param img_shape: size of input image as (W, H)
    :type img_shape: Tuple[int, int]
    :param output_shape: output shape of the neural net
    :type output_shape: tuple
    """
    def __init__(self, img_shape: Tuple[int, int], output_shape: tuple):
        super(SimpleConvNetModel, self).__init__()
        # TODO: Set up Conv Net of your choosing. You can / should hardcode
        # the sizes and layers of this Neural Net. The img_size tells you what
        # the input size should be and you have to determine the best way to
        # represent the output_shape (tuple of 2 ints, tuple of 1 int, just an
        # int , etc).
        raise NotImplementedError()


    def forward(x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        raise NotImplementedError()


if __name__ == "__main__":
    neural_net = SimpleNeuralNetModel([400,200,10])
    print(neural_net.linears)
    train_load, val_load, test_load = get_data_loaders('train/train_samples.pkl',
                                                       'train/correct_train_labels.pkl',
                                                       'val/val_samples.pkl',
                                                       'val/correct_val_labels.pkl',
                                                       'test/test_samples.pkl',
                                                       'test/correct_test_labels.pkl')

    model = SimpleNeuralNetModel(layer_sizes=[784, 200, 200, 10])
    optimizer = optim.Adam(params=model.parameters(), lr = 0.01)
    model.train()
    for t in range(100):
        for batch_index, (input_t, y) in enumerate(train_load):
            optimizer.zero_grad()
            preds = model(input_t)
            loss = model.loss(preds, y.view(1,len(y)))
            loss.backward()
            optimizer.step()

