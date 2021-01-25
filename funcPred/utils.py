import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.io import arff
import pandas as pd

torch.set_default_dtype(torch.double)

def fit(loss, params, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None):
    """
    Arguments:
        loss: given x and y in batched form, evaluates loss.
        params: list of parameters to optimize.
        X: input data, torch tensor.
        Y: output data, torch tensor.
        Xval: input validation data, torch tensor.
        Yval: output validation data, torch tensor.
    """

    train_dset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    opt = opt(params, **opt_kwargs)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        with torch.no_grad():
            val_losses.append(loss(Xval, Yval).item())
        if verbose:
            print("----- epoch %03d / %03d | %3.5f" % (epoch + 1, epochs, val_losses[-1]))
        batch = 1
        train_losses.append([])
        for Xbatch, Ybatch in train_loader:
            opt.zero_grad()
            l = loss(Xbatch, Ybatch)
            l.backward()
            opt.step()
            train_losses[-1].append(l.item())
            if verbose:
                print("batch %03d / %03d | %3.5f" %
                      (batch, len(train_loader), np.mean(train_losses[-1])))
            batch += 1
            if callback is not None:
                callback()
    return val_losses, train_losses

def extract_data_from_file(path):
    with open(path) as file:
        samples = file.read().splitlines()
    samples = np.array([sample.split(",") for sample in samples]).astype(np.float)
    np.random.shuffle(samples)
    X = samples[:, :-1]
    Y = np.ndarray.flatten(samples[:, -1])
    Y[Y == 0] = -1
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return torch.from_numpy(X), torch.from_numpy(Y)

def extract_data_from_file_arff(path):
    data, meta = arff.loadarff(path)
    return data, meta


def get_data(N, Xdim, w_true, b_true):
    X = torch.randn(N, Xdim)
    Y = X @ w_true + b_true + torch.randn(N)
    return X, Y

def evaluate_model(Y, Y_pred):
    num = len(Y)
    temp = Y - Y_pred
    acc = 100*len(temp[temp == 0]) / num
    print("accuracy: {}%".format(acc))


# data, meta = extract_data_from_file_arff(r"C:\Users\sagil\Desktop\funcPred\tip_spam_data\IS_journal_tip_spam.arff")
# df = pd.DataFrame(data)
# df.head()


