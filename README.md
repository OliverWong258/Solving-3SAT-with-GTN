This is an adjustment of the GTN model from [SAT-Solver-usingNNs](https://github.com/tatiana-boura/SAT-Solver-using-NNs/tree/main), primarily involving adjustments in the number of attention heads in the self-attention mechanism and modifications to the activation function of the neural network. These changes aim to enhance the model's performance, especially in scenarios where there is a significant difference in the complexity of training and testing data.

To run the program, enter `python main.py --m <model_path> --s <separate>` in the command line.

Here, `<model_path>` is the name of the saved model, and `<separate>` indicates whether to conduct separate testing, with 0 for no and 1 for yes.

Other optional parameters include:
- `--d` Path to training data, default is `./data`
- `--e` Model embedding dimension
- `--h` Number of attention heads in the model(the default is 2)
- `--l` Number of layers in the model
- `--r` Dropout rate
- `--ls` Neuron density of the final linear layer
- `--b` Batch size

The output model files are saved in the `./models` directory, and the learning curves are saved in the `./plots` directory.
