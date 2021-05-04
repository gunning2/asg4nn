from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim import Adam


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = torch.nn.MSELoss()
    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    # print(min_loss)

    optimizer = Adam(model.parameters(), lr = 0.001)
    prev_loss = float('inf')
    for epoch_i in tqdm(range(no_epochs)):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            x, y = sample['input'], sample['label']
            optimizer.zero_grad()
            y_hat = model.forward(x)
            # loss = loss_function(y_hat.unsqueeze(dim=0), y.long())
            loss = loss_function(y_hat.reshape(1).float(), y.float())
            loss.backward()
            optimizer.step()
        total_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        if total_loss < prev_loss:
            torch.save(model.state_dict(), "saved/saved_model.pkl", _use_new_zipfile_serialization=False)
            prev_loss = total_loss
        print()
        print(total_loss)
        losses.append(model.evaluate(model, data_loaders.test_loader, loss_function))
        # print('e')
    torch.save(model.state_dict(), "saved/saved_model.pkl", _use_new_zipfile_serialization=False)
    # print(losses)


if __name__ == '__main__':
    no_epochs = 80
    train_model(no_epochs)

