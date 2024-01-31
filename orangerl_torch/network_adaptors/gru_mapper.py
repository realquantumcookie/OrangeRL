from .rnn_mapper import RNNNetworkAdaptor

"""
Since GRU implementation in PyTorch is the same as RNN, we can just use the RNNNetworkAdaptor.
Note that the GRU layers must be specified with batch_first=True.
"""
GRUNetworkAdaptor = RNNNetworkAdaptor