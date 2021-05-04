import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()
        input_size=6
        hidden_size=20
        hidden_2_size=26
        output_size=1
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.h1_h2 = nn.Linear(hidden_size, hidden_2_size)
        
        self.nonlinear_activation = nn.RReLU()
        self.nonlinear_activation2 = nn.Sigmoid()
        self.hidden_to_output = nn.Linear(hidden_2_size, output_size)
        # Relu here? sigmoid? relu seems better, was never actually activated just realized oops
        #  Current config is best so far, change loss
        self.output_activation = nn.Softmax(dim=0)
        # nn.soft

    def forward(self, network_input):
        network_input = network_input.float()
        hidden1 = self.input_to_hidden(network_input)
        hidden1 = self.nonlinear_activation(hidden1)
        hidden2 = self.h1_h2(hidden1)
        hidden2 = self.nonlinear_activation2(hidden2)
        network_output = self.hidden_to_output(hidden2)
        # network_output = self.output_activation(network_output)
        # print(network_output)
        return network_output.reshape(1).float()


    def evaluate(self, model, test_loader, loss_function):
        loss = 0
        for idx, sample in enumerate(test_loader): 
            x, y = sample['input'], sample['label']
            y_hat = model.forward(x)
            # For MSE
            loss += loss_function(y_hat.reshape(1).float(), y.float())
        return float(loss)

def main():
    # model = Action_Conditioned_FF()
    pass

if __name__ == '__main__':
    main()
