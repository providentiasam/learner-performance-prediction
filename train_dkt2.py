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

import traceback


def train(
    train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, device
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

    train_batches = prepare_batches(train_data, batch_size)
    val_batches = prepare_batches(val_data, batch_size)

    for epoch in range(num_epochs):
        # Training
        for (
            item_inputs,
            skill_inputs,
            label_inputs,
            item_ids,
            skill_ids,
            labels,
        ) in train_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)

            loss = compute_loss(preds, labels.to(device), criterion)
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
                item_inputs = item_inputs.to(device)
                skill_inputs = skill_inputs.to(device)
                label_inputs = label_inputs.to(device)
                item_ids = item_ids.to(device)
                skill_ids = skill_ids.to(device)
                preds = model(
                    item_inputs, skill_inputs, label_inputs, item_ids, skill_ids
                )
                val_auc = compute_auc(torch.sigmoid(preds).cpu(), labels.cpu())
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
    parser.add_argument("--dataset", type=str, default="ednet_small")
    parser.add_argument("--logdir", type=str, default="runs/dkt")
    parser.add_argument("--savedir", type=str, default="save/dkt")
    parser.add_argument("--hid_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--num_hid_layers", type=int, default=1)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seqlen", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default='bt_dkt2')
    parser.add_argument("--name", type=str, default="train_dkt2")
    parser.add_argument("--verbose_error", type=bool, default=True)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_random_seeds(args.seed)

    # initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project, name=args.name, config=args)
        print('wandb init')
    else:
        args.project = None

    full_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data.csv"), sep="\t"
    )
    train_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_train.csv"), sep="\t"
    )
    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    )

    # train_data, val_data = get_chunked_data(train_df, max_length=args.seqlen, train_split=0.8, \
    #     randomize=True, stride=args.seqlen//2, non_overlap_only=True)
    train_data, val_data = get_data(train_df, train_split=0.8)

    model = DKT2(
        int(full_df["item_id"].max()),
        int(full_df["skill_id"].max()),
        args.hid_size,
        args.embed_size,
        args.num_hid_layers,
        args.drop_prob,
    )
    model.to(args.device)
    if torch.cuda.device_count() > 1:
        print('using {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        try:
            # Train
            param_str = f"{args.dataset}"
            logger = Logger(
                os.path.join(args.logdir, param_str),
                project_name=args.project,
                run_name=args.name,
            )
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
                args.device,
            )
            break
        except RuntimeError as e:
            if args.verbose_error:
                print("error detail")
                traceback.print_exc()
            args.batch_size = args.batch_size // 2
            print(f"Batch does not fit on gpu, reducing size to {args.batch_size}")

    logger.close()

    model = saver.load()
    model.to(args.device)
    # test_data, _ = get_chunked_data(test_df, max_length=args.seqlen, train_split=1.0, randomize=False)
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