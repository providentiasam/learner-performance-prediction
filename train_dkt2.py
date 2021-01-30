import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from models.model_dkt2 import DKT2
from utils import *
from train_utils import *


def train(
    train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size
):
    """Train DKT model.

    Arguments:
        train_data (list of lists of torch Tensor)
        val_data (list of lists of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for (
            item_inputs,
            skill_inputs,
            label_inputs,
            item_ids,
            skill_ids,
            labels,
        ) in train_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)

            loss = compute_loss(preds, labels.cuda(), criterion)
            train_auc = compute_auc(torch.sigmoid(preds).detach().cpu(), labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            metrics.store({"loss/train": loss.item()})
            metrics.store({"auc/train": train_auc})

            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)

        # Validation
        model.eval()
        for (
            item_inputs,
            skill_inputs,
            label_inputs,
            item_ids,
            skill_ids,
            labels,
        ) in val_batches:
            with torch.no_grad():
                item_inputs = item_inputs.cuda()
                skill_inputs = skill_inputs.cuda()
                label_inputs = label_inputs.cuda()
                item_ids = item_ids.cuda()
                skill_ids = skill_ids.cuda()
                preds = model(
                    item_inputs, skill_inputs, label_inputs, item_ids, skill_ids
                )
            val_auc = compute_auc(torch.sigmoid(preds).cpu(), labels)
            metrics.store({"auc/val": val_auc})
        model.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics["auc/val"], model)
        if stop:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DKT.")
    parser.add_argument("--dataset", type=str, default="ednet_medium")
    parser.add_argument("--logdir", type=str, default="runs/dkt")
    parser.add_argument("--savedir", type=str, default="save/dkt")
    parser.add_argument("--hid_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--num_hid_layers", type=int, default=1)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
    
    args = parser.parse_args()

    set_random_seeds(args.seed)

    full_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data.csv"), sep="\t"
    )
    train_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_train.csv"), sep="\t"
    )
    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    )

    train_data, val_data = get_data(train_df, train_split=0.8)

    model = DKT2(
        int(full_df["item_id"].max()),
        int(full_df["skill_id"].max()),
        args.hid_size,
        args.embed_size,
        args.num_hid_layers,
        args.drop_prob,
    )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = nn.DataParallel(model)
    # model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        try:
            # Train
            param_str = f"{args.dataset}"
            logger = Logger(os.path.join(args.logdir, param_str))
            saver = Saver(args.savedir, param_str)
            train(
                train_data,
                val_data,
                model,
                optimizer,
                logger,
                saver,
                args.num_epochs,
                args.batch_size,
            )
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f"Batch does not fit on gpu, reducing size to {args.batch_size}")

    logger.close()

    model = saver.load()
    test_data, _ = get_data(test_df, train_split=1.0, randomize=False)
    test_batches = prepare_batches(test_data, args.batch_size, randomize=False)

    # Predict on test set
    test_preds = eval_batches(model, test_batches, device='cuda')

    # Write predictions to csv
    if 0:
        test_df["DKT2"] = test_preds
        test_df.to_csv(
            f"data/{args.dataset}/preprocessed_data_test.csv", sep="\t", index=False
        )
    test_df['model_pred'] = test_preds
    print("auc_test = ", roc_auc_score(test_df["correct"], test_preds))
    print("acc_test = ", (test_df['correct'] == test_df['model_pred'].round()).describe())