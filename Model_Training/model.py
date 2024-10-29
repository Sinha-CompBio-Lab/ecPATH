#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from utils import *

## check available device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print("device:", device)


##================================================================================================
class MLP_regression(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init):
        super(MLP_regression, self).__init__()

        self.layer0 = nn.Sequential(
            # nn.Conv1d(in_channels=n_inputs, out_channels=n_hiddens,kernel_size=1, stride=1, bias=True),
            nn.Linear(n_inputs, n_hiddens),
            # nn.ReLU(),  ## 2020.03.26: for positive gene expression
            nn.Dropout(dropout),
        )

        # self.layer1 = nn.Conv1d(in_channels=n_hiddens, out_channels=n_outputs,kernel_size=1, stride=1, bias=True)
        self.layer1 = nn.Linear(n_hiddens, n_outputs)

        ## ---- set bias of the last layer ----
        if bias_init is not None:
            self.layer1.bias = bias_init

    ##-------------------------------------
    def forward(self, x):

        # print("x.shape - input of forward:", x.shape)    ## [n_tiles,512]
        x = self.layer0(x)
        x = self.layer1(x)

        # print("x.shape - before mean:", x.shape)        ## [n_tiles,512]

        x = torch.mean(x, dim=0)  ## sum over tiles
        # print("x.shape -- after mean:", x.shape)        ## [n_genes]

        return x  ## predicted gene_values [n_genes]


##================================================================================================
def training_epoch(model, optimizer, train_set, batch_size):
    model.train()
    loss_fn = nn.MSELoss()

    n_slides_train = len(train_set)
    # print("n_slides_train:", n_slides_train)

    ## shuffle training set
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)

    loss_list = []
    labels = []
    preds = []
    for i_batch in range(0, n_slides_train, batch_size):
        # print(i_slide)

        n_slides_batch = min(batch_size, n_slides_train - i_batch)
        # print(n_slides_batch)

        ##---------------------------
        ## for each batch
        loss = 0
        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]

            x, y = train_set[idx]

            # print(x.shape)  ## [512, n_tiles]
            # print(y.shape)  ## [n_genes]

            pred = model(x.float().to(device))
            # print("pred.shape:", pred.shape)   ## [n_genes]

            loss += loss_fn(pred, y.float().to(device))

            labels.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

        loss /= n_slides_batch

        loss_list += [loss.detach().cpu().numpy()]  ## add loss of each batch to a list

        ## reset gradients to zero
        optimizer.zero_grad()

        ## compute gradients
        loss.backward()

        ## update parameters using gradients
        optimizer.step()

    labels = np.array(labels)
    preds = np.array(preds)

    coef, slope = compute_coef_slope(labels, preds)

    # print(f"loss: {np.mean(loss_list)}, coef: {np.mean(coef)}, slope:, {np.mean(slope)}")

    return np.mean(loss_list), np.mean(coef), np.mean(slope)


##================================================================================================
def predict(model, valid_set):
    model.eval()
    loss_fn = nn.MSELoss()

    labels = []
    preds = []
    loss_list = []

    with torch.no_grad():
        for x, y in valid_set:
            pred = model(x.float().to(device))  ## y_pred = model(x)

            loss = loss_fn(pred, y.float().to(device))
            loss_list += [loss.detach().cpu().numpy()]  ## convert to numpy
            # pred = nn.ReLU()(pred)                    ## y_pred ## 2026.03.26: for negative gene expression

            labels.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

    ## convert list to 2D array
    labels = np.array(labels)
    preds = np.array(preds)

    coef, slope = compute_coef_slope(labels, preds)

    return np.mean(loss_list), np.mean(coef), np.mean(slope), labels, preds


##================================================================================================
def fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size):

    train_loss_list = []
    train_coef_list = []
    train_slope_list = []

    valid_loss_list = []
    valid_coef_list = []
    valid_slope_list = []

    epoch_since_best = 0
    valid_coef_old = -1.0

    for e in range(max_epochs):
        epoch_since_best += 1

        ## train
        train_loss, train_coef, train_slope = training_epoch(
            model, optimizer, train_set, batch_size
        )

        ## predict
        valid_loss, valid_coef, valid_slope, valid_labels, valid_preds = predict(
            model, valid_set
        )

        print(
            f"{e}, train_loss: {train_loss:.4f}, coef: {train_coef:.4f}, slope: {train_slope:.4f},\
            valid_loss: {valid_loss:.4f}, coef: {valid_coef:.4f}, slope: {valid_slope:.4f}"
        )

        train_loss_list.append(train_loss)
        train_coef_list.append(train_coef)
        train_slope_list.append(train_slope)

        valid_loss_list.append(valid_loss)
        valid_coef_list.append(valid_coef)
        valid_slope_list.append(valid_slope)

        if valid_coef > valid_coef_old:
            epoch_since_best = 0
            valid_coef_old = valid_coef

        if epoch_since_best == patience:
            print("Early stopping at epoch {}".format(e + 1))
            break

    return (
        model,
        train_loss_list,
        train_coef_list,
        train_slope_list,
        valid_loss_list,
        valid_coef_list,
        valid_slope_list,
        valid_labels,
        valid_preds,
    )


##================================================================================================
def analyze_result(
    result_dir,
    genes,
    model,
    train_loss,
    train_coef,
    train_slope,
    valid_loss,
    valid_coef,
    valid_slope,
    valid_labels,
    valid_preds,
    test_set,
):

    ## save trained model
    torch.save(model.state_dict(), f"{result_dir}model_trained.pth")

    train_valid_loss = np.array(
        (train_loss, valid_loss, train_coef, valid_coef, train_slope, valid_slope)
    ).T
    np.savetxt(f"{result_dir}train_valid_loss.txt", train_valid_loss, fmt="%.6f")

    ## predict test
    test_loss, test_coef, test_slope, test_labels, test_preds = predict(model, test_set)

    valid_coef1, valid_slope1 = compute_coef_slope(valid_labels, valid_preds)
    test_coef1, test_slope1 = compute_coef_slope(test_labels, test_preds)

    np.savetxt(
        f"{result_dir}coef_slope.txt",
        np.array((valid_coef1, test_coef1, valid_slope1, test_slope1)).T,
        fmt="%8s",
    )

    ## sorted based on valid test
    i1 = np.argsort(valid_coef1)[::-1]
    np.savetxt(
        f"{result_dir}coef_sorted_based_valid.txt",
        np.array(
            (
                i1,
                genes[i1],
                valid_coef1[i1],
                test_coef1[i1],
                valid_slope1[i1],
                test_slope1[i1],
            )
        ).T,
        fmt="%8s %22s %8s %8s %8s %8s",
    )

    ## sorted based on test set
    i2 = np.argsort(test_coef1)[::-1]
    np.savetxt(
        f"{result_dir}coef_sorted_based_test.txt",
        np.array(
            (
                i2,
                genes[i2],
                valid_coef1[i2],
                test_coef1[i2],
                valid_slope1[i2],
                test_slope1[i2],
            )
        ).T,
        fmt="%8s %22s %8s %8s %8s %8s",
    )

    # np.savetxt(f"{result_dir}valid_labels.txt", valid_labels, fmt="%.8f")
    # np.savetxt(f"{result_dir}valid_preds.txt", valid_preds, fmt="%.8f")
    np.save(f"{result_dir}test_labels.npy", test_labels)
    np.save(f"{result_dir}test_preds.npy", test_preds)

    ##-----------------------------------------------------------------------
    nx, ny = 2, 3
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 4, ny * 3))

    ## 1st col:
    ax[0, 0].plot(train_loss, "k--", label="train")
    ax[0, 0].plot(valid_loss, "b-", label="valid")

    ax[1, 0].plot(train_coef, "k--", label="train")
    ax[1, 0].plot(valid_coef, "b-", label="valid")

    ax[2, 0].plot(train_slope, "k--", label="train")
    ax[2, 0].plot(valid_slope, "b-", label="valid")

    for j in range(ny):
        ax[j, 0].set_xlabel("n_epochs")
        ax[j, 0].legend()

    ax[0, 0].set_ylabel("loss")
    ax[1, 0].set_ylabel("coef")
    ax[2, 0].set_ylabel("slope")

    ## 2nd col:
    i = 0
    label = test_labels[:, i2[i]]
    pred = test_preds[:, i2[i]]

    ax[0, 1].plot(label, pred, "o", label="test")
    ax[0, 1].plot(
        [min(min(label), min(pred)), max(max(label), max(pred))],
        [min(min(label), min(pred)), max(max(label), max(pred))],
        "--",
    )
    ax[0, 1].set_xlabel("label")
    ax[0, 1].set_ylabel("pred")
    ax[0, 1].set_title(
        f"{genes[i2[i]]}, R={test_coef1[i2[i]]:3.2f}, slope={test_slope1[i2[i]]:3.2f}"
    )

    ## coef histogram
    bins = np.linspace(min(test_coef1), max(test_coef1), 10, endpoint=False)
    ax[1, 1].hist(test_coef1, bins, histtype="bar", rwidth=0.8)
    ax[1, 1].set_xlabel("coef")
    ax[1, 1].set_ylabel("#genes")

    ## test
    bins = np.linspace(min(test_slope1), max(test_slope1), 10, endpoint=False)
    ax[2, 1].hist(test_slope1, bins, histtype="bar", rwidth=0.8)
    ax[2, 1].set_xlabel("slope")
    ax[2, 1].set_ylabel("#genes")

    plt.tight_layout(h_pad=1, w_pad=0.5)
    plt.savefig(f"{result_dir}loss.pdf", format="pdf", dpi=50)

    ##---------------------------

    ## best predictable genes
    nx, ny = 5, 2
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 4, ny * 3))

    ij = 0
    for j in range(ny):
        for i in range(nx):
            ## test set:
            label = test_labels[:, i2[ij]]
            pred = test_preds[:, i2[ij]]

            ax[j, i].plot(label, pred, "o", label="test")
            ax[j, i].plot(
                [min(min(label), min(pred)), max(max(label), max(pred))],
                [min(min(label), min(pred)), max(max(label), max(pred))],
                "--",
            )

            ax[j, i].set_xlabel("label")
            ax[j, i].set_ylabel("pred")
            ax[j, i].set_title(
                f"{genes[i2[ij]]}, R={test_coef1[i2[ij]]:3.2f}, slope={test_slope1[i2[ij]]:3.2f}"
            )

            ij += 1
        # ax[0,0].legend()

    plt.tight_layout(h_pad=1, w_pad=0.5)
    plt.savefig(f"{result_dir}best_preds.pdf", format="pdf", dpi=50)


##================================================================================================
