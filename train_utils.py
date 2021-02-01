import torch
import numpy as np
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score


def get_preds(preds, item_ids, skill_ids, labels):
    preds = preds[labels >= 0]

    # if item_ids is not None:
    #     item_ids = item_ids[labels >= 0]
    #     preds = preds[torch.arange(preds.size(0)), item_ids]
    # elif skill_ids is not None:
    #     skill_ids = skill_ids[labels >= 0]
    #     preds = preds[torch.arange(preds.size(0)), skill_ids]

    return preds


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


def get_data(df, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        train_split (float): proportion of data to use for training
    """
    item_ids = [
        torch.tensor(u_df["item_id"].values, dtype=torch.long)
        for _, u_df in df.groupby("user_id")
    ]
    skill_ids = [
        torch.tensor(u_df["skill_id"].values, dtype=torch.long)
        for _, u_df in df.groupby("user_id")
    ]
    labels = [
        torch.tensor(u_df["correct"].values, dtype=torch.long)
        for _, u_df in df.groupby("user_id")
    ]

    item_inputs = [
        torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids
    ]
    skill_inputs = [
        torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids
    ]
    label_inputs = [
        torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels
    ]

    data = list(
        zip(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    )
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def get_chunked_data(df, max_length=200, train_split=0.8, randomize=False, stride=None, non_overlap_only=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """
    item_ids = [
        torch.tensor(u_df["item_id"].values, dtype=torch.long)
        for _, u_df in df.groupby("user_id")
    ]
    skill_ids = [
        torch.tensor(u_df["skill_id"].values, dtype=torch.long)
        for _, u_df in df.groupby("user_id")
    ]
    labels = [
        torch.tensor(u_df["correct"].values, dtype=torch.long)
        for _, u_df in df.groupby("user_id")
    ]
    stride = max_length if stride is None else stride

    item_inputs = [
        torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids
    ]
    skill_inputs = [
        torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids
    ]
    label_inputs = [
        torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels
    ]

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
        batch = data[k: k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [
            pad_sequence(seqs, batch_first=True, padding_value=0)
            if (seqs[0] is not None)
            else None
            for seqs in seq_lists[:-1]
        ]
        labels = pad_sequence(
            seq_lists[-1], batch_first=True, padding_value=-1
        )  # Pad labels with -1
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


def eval_batches(model, batches, device='cpu', is_dkt1=False):
    model.eval()
    test_preds = np.empty(0)
    for (
            item_inputs,
            skill_inputs,
            label_inputs,
            item_ids,
            skill_ids,
            labels,
    ) in batches:
        with torch.no_grad():
            if device == 'cuda':
                item_inputs = item_inputs.cuda()
                skill_inputs = skill_inputs.cuda()
                label_inputs = label_inputs.cuda()
                item_ids = item_ids.cuda()
                skill_ids = skill_ids.cuda()
                labels = labels.cuda()
            if is_dkt1:
                preds, _ = model(item_inputs, skill_inputs)
                preds = torch.sigmoid(
                    get_preds(preds, item_ids, skill_ids, labels)
                ).cpu().numpy()
            else:
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])
    return test_preds
