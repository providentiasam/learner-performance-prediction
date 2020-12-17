import argparse
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from models.model_sakt2 import SAKT
from utils import *
import os


def window_split(x, window_size=100, stride=50, return_nonoverlap=False):
    if len(x) <= window_size:
        return [0] if return_nonoverlap else [x]
    length = x.size(0)
    splits = []
    non_overlap_from = []
    prev_slice_end = 0
    for slice_start in range(0, length + 1, stride):
        slice_end = min(slice_start + window_size, length)
        if slice_end > slice_start:
            splits.append(x[slice_start:slice_end])
            non_overlap_from.append(prev_slice_end - slice_start)
            prev_slice_end = slice_end
    if return_nonoverlap:
        return non_overlap_from
    else:
        return splits


def get_data(df, max_length, train_split=0.8, randomize=False, stride=None, non_overlap_only=True):
    """Extract sequences from dataframe.
    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long) for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long) for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long) for _, u_df in df.groupby("user_id")]
    stride = max_length if stride is None or np.isnan(stride) else stride

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list, stride):
        if list[0] is None:
            return list
        list = [window_split(elem, max_length, stride) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l, stride) for l in lists]
    if non_overlap_only:
        non_overlap_from = [y for x in labels for y in \
            window_split(x, window_size=max_length, stride=stride, return_nonoverlap=True)]
        # chunked_lists.append([y for x in item_inputs for y in window_split(x, max_length, stride)[1]])
        non_overlap_labels = []
        for org_label, index_begin in zip(chunked_lists[5], non_overlap_from):
            label_seq = org_label.detach().clone()
            label_seq[:index_begin] = -1
            non_overlap_labels.append(label_seq)
        chunked_lists = chunked_lists[:5] + [non_overlap_labels]
    
    data = list(zip(*chunked_lists))
    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    if randomize:
        shuffle(train_data)
        shuffle(val_data)
    return train_data, val_data


def prepare_batches(data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch
    Output:
        batches (list of lists of torch Tensor)
    """
    if randomize:
        shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, grad_clip):
    """Train SAKT model.

    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif optimizer == 'noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        def noam(step: int):
            step = max(1, step)
            warmup_steps = 2000
            scale = warmup_steps ** 0.5 * min(
                step ** (-0.5), step * warmup_steps ** (-1.5))
            return scale
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam)
    else:
        raise NotImplementedError

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size, randomize=False)

        # Training
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            loss = compute_loss(preds, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            if scheduler is not None:
                metrics.store({'lr': scheduler.get_lr()[0]})
                scheduler.step()
            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)

        # Validation
        model.eval()
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds).cpu()
            val_auc = compute_auc(preds, labels)
            metrics.store({'auc/val': val_auc})
        
        model.train()

        # Save model
        average_metrics = metrics.average()
        average_metrics['epoch'] = epoch
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAKT.')
    parser.add_argument('--setup', type=str, default='ednet_small')
    args_ = parser.parse_args()
    DEBUGGING = True
    if DEBUGGING:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  
        setup_path = './setups/sakt_loop_test.xlsx'
        setup_page = pd.DataFrame([{
            'dataset': 'ednet',
            'num_attn_layers': 3,
            'max_length': 200,
            'embed_size': 50,
            'num_heads': 5,
            'encode_pos': 1, 'max_pos': 10, 'drop_prob': 0.2, 'batch_size': 500, 'optimizer': 'noam',
            'lr': 0.003, 'grad_clip': 10, 'num_epochs': 500, 'repeat': 3, 'stride': 100, 'dim_ff': 30
        }])
    else:
        setup_path = './setups/sakt_loop_{}.xlsx'.format(args_.setup)
        setup_page = pd.read_excel(setup_path)
    
  
    result_cols = ['test1', 'test2', 'test3', 'valid1', 'valid2', 'valid3', 'logdir', 'savedir']
    setup_cols = [x for x in setup_page.columns if x not in result_cols]
    dataset = args_.setup if 'dataset' not in setup_cols else setup_page['dataset'].iloc[0]
    for col in result_cols:
        if col not in setup_page.columns:
            setup_page[col] = np.nan    
    setup_page[setup_cols] = setup_page[setup_cols].ffill()
    
    full_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")

    for setup_index in setup_page.index:
        args = setup_page.loc[setup_index]
        setup_page.loc[setup_index, 'logdir'] = 'runs/sakt/' + dataset + '/'
        setup_page.loc[setup_index, 'savedir'] = 'save/sakt/' + dataset + '/'
        args = setup_page.loc[setup_index]
        args.loc['dataset'] = dataset
        print(args)
        stop_experiment = False # Stop current setup for whatever reason possible.
        if args[['test1', 'test2', 'test3']].notnull().all():
            print(args, ' already done')
            continue
        for rand_seed in range(int(args['repeat'])):
            set_random_seeds(rand_seed)
            train_data, val_data = get_data(train_df, int(args.max_length), randomize=True, \
                stride=int(args.stride), non_overlap_only=True)
            num_items = int(full_df["item_id"].max() + 1)
            num_skills = int(full_df["skill_id"].max() + 1)
            model = SAKT(num_items, num_skills, int(args.embed_size), int(args.num_attn_layers), int(args.num_heads),
                        bool(args.encode_pos), int(args.max_pos), args.drop_prob).cuda()
            if torch.cuda.device_count() > 1:
                print('using {} GPUs'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(torch.device("cuda"))

            while True: # Reduce batch size until it fits on GPU
                try:
                    # Train
                    param_str = '_'.join([str(x) + str(y) for x, y in args.to_dict().items()])[:200]
                    optimizer = 'adam' if 'optimizer' not in args.index else args['optimizer']
                    logger = Logger(os.path.join(args.logdir, param_str))
                    saver = Saver(args.savedir, param_str, patience=7 if dataset not in {'ednet', 'ednet_medium'} else 3)
                    train(train_data, val_data, model, optimizer, logger, saver, int(args.num_epochs),
                        int(args.batch_size), args.grad_clip)
                    break
                except RuntimeError:
                    args.loc['batch_size'] = args.batch_size // 2
                    print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')
                    if args.batch_size < 25:
                        stop_experiment = True
                        break
            if stop_experiment:
                print('GPU too small to create meaningfully large mini-batch.')
                break
            model = saver.load()

            if 1:
                test_data, _ = get_data(test_df, int(args.max_length), train_split=1.0, \
                    randomize=False, stride=1, non_overlap_only=True)
                test_batches = prepare_batches(test_data, int(args.batch_size), randomize=False)
                test_preds = np.empty(0)
                model.eval()
                for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
                    item_inputs = item_inputs.cuda()
                    skill_inputs = skill_inputs.cuda()
                    label_inputs = label_inputs.cuda()
                    item_ids = item_ids.cuda()
                    skill_ids = skill_ids.cuda()
                    with torch.no_grad():
                        preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                        preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
                        test_preds = np.concatenate([test_preds, preds])
                setup_score = roc_auc_score(test_df['correct'], test_preds)
                print(setup_score)
                logger.log_scalars({'test auc': setup_score}, step=0)
                exp_ind = rand_seed + 1
                setup_page.loc[setup_index, 'test{}'.format(exp_ind)] = setup_score
                setup_page.loc[setup_index, 'valid{}'.format(exp_ind)] = saver.score_max
                setup_page.to_excel(setup_path.replace('setups/', 'results/'))

            logger.close()
            del model

