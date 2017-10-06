Inplementation of the same model with different Deep Learning frameworks, namely TensorFlow, PyTorch and Keras (TensorFlow backend) to the CIFAR-10 dataset. The objective was not to get the best score, but to have exactly the same model in the different frameworks.

Model used:

[conv-relu-batchnorm-MaxPool] x n -> [affine-relu-dropout] -> [affine] -> [softmax]

I find that n = 3 provides the best results.

