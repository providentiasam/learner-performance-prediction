import torch
import pandas as pd
from copy import deepcopy
import random
import numpy as np



def test_perturbation(bt_test_df, diff_threshold=0):
    bt_test_df.loc[:, 'testpass'] = np.nan
    bt_test_df.loc[:, 'model_diff'] = np.nan
    bt_test_df['new_idx'] = bt_test_df.index
    stacked_df_dict = {}
    for orig_user_id, group in bt_test_df.groupby('orig_user_id'):
        new_group = group.set_index(['user_id', 'orig_idx'], drop=True).unstack('user_id')
        unperturbed_user_id = new_group['is_perturbed'].fillna(0.0).abs().sum().idxmin()
        assert new_group['is_perturbed'][unperturbed_user_id].fillna(0).sum() == 0
        model_diff = new_group['model_pred'].subtract(new_group['model_pred'][unperturbed_user_id], axis=0)
        for virtual_user_id in model_diff.columns:
            new_group[('model_diff', virtual_user_id)] = model_diff[virtual_user_id].copy()
        stacked_df_dict[orig_user_id] = new_group
    result_df = pd.concat([y.stack('user_id') for _, y in stacked_df_dict.items()], axis=0).reset_index(drop=False)
    result_df = result_df.set_index('new_idx').sort_index()
    result_df.index = result_df.index.astype(int)
    perturbed_idx = result_df['is_perturbed'] != 0
    result_df.loc[perturbed_idx, 'testpass'] = \
        (result_df.loc[perturbed_idx, 'model_diff'] * result_df.loc[perturbed_idx, 'is_perturbed']) >= -diff_threshold
    groupby_key = ['all', 'is_perturbed']
    return result_df, groupby_key


def gen_perturbation(orig_df, perturb_func, **pf_args):
    """
    Generates perturbed pandas dataframe object.

    Arguments:
        orig_df: original pandas dataframe object
        perturb_func: perturbation function (ex. replace, add, ...)
        pf_args: additional arguments for perturb_func
    """
    new_df_list = []
    for user_id, user_key_df in orig_df.groupby(["user_id"]):
        new_df = perturb_func(user_key_df, **pf_args)
        new_df_list.append(new_df)
    new_data = pd.concat(new_df_list, axis=0).reset_index(drop=True)
    new_data['user_id'] = new_data['user_id'].astype(str)
    data_meta = {
        'num_sample': new_data['user_id'].unique().shape[0],
        'num_interaction': new_data.shape[0],
    }
    virtual_user_id = 0
    for user_id in new_data['user_id'].unique():
        new_data.loc[new_data['user_id']==user_id, 'user_id'] = virtual_user_id
        virtual_user_id += 1
    return new_data, data_meta


def perturb_review_step(orig_df):
    correct_df = orig_df.loc[orig_df["correct"] == 1]
    incorrect_df = orig_df.loc[orig_df["correct"] == 0]
    review_df = deepcopy(incorrect_df)
    review_df["correct"] = 1
    orig_df = orig_df.append(review_df).reset_index(drop=True)
    return orig_df


def perturb_add_last(orig_df, row_index, new_value):
    new_df = deepcopy(orig_df.iloc[[row_index]])
    new_df.loc[:, "correct"] = new_value
    orig_df = orig_df.append(new_df).reset_index(drop=True)
    return orig_df


def perturb_add_last_random(orig_df):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'is_perturbed'] = 0
    row_index = random.randrange(0, len(orig_df))
    corr_df = perturb_add_last(orig_df, row_index, 1)
    corr_df.loc[:, 'user_id'] = corr_df['user_id'].astype(str) + "corr"
    corr_df.loc[:, 'is_perturbed'] = 1
    incorr_df = perturb_add_last(orig_df, row_index, 0)
    incorr_df.loc[:, 'user_id'] = incorr_df['user_id'].astype(str) + "incorr"
    incorr_df.loc[:, 'is_perturbed'] = -1

    orig_df = orig_df.append(orig_df.iloc[row_index]).reset_index(drop=True)
    corr_df = corr_df.append(corr_df.iloc[row_index]).reset_index(drop=True)
    incorr_df = incorr_df.append(incorr_df.iloc[row_index]).reset_index(drop=True)

    new_df_list = [orig_df, corr_df, incorr_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)


def perturb_insertion(orig_df, copy_idx, insert_idx, corr_value):
    new_df = deepcopy(orig_df.iloc[[copy_idx]])
    new_df.loc[:, "correct"] = corr_value
    new_df.loc[:, 'orig_idx'] = -1
    orig_df = orig_df.iloc[:insert_idx].append(new_df)\
        .append(orig_df.iloc[insert_idx:]).reset_index(drop=True)
    return orig_df


def perturb_insertion_random(orig_df, insert_policy=None):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'orig_idx'] = orig_df.index
    orig_df.loc[:, 'is_perturbed'] = 0
    copy_idx = random.randrange(0, len(orig_df))
    if insert_policy == "first":
        insert_idx = 0
    elif insert_policy == "middle":
        insert_idx = len(orig_df) // 2
    elif insert_policy == "last":
        insert_idx = len(orig_df) - 1
    else:
        insert_idx = random.randrange(0, len(orig_df))
    corr_df = perturb_insertion(orig_df, copy_idx, insert_idx, 1)
    corr_df.loc[:, 'user_id'] = corr_df['user_id'].astype(str) + "_corr"
    corr_df.loc[insert_idx+1:, 'is_perturbed'] = 1
    incorr_df = perturb_insertion(orig_df, copy_idx, insert_idx, 0)
    incorr_df.loc[:, 'user_id'] = incorr_df['user_id'].astype(str) + "_incorr"
    incorr_df.loc[insert_idx+1:, 'is_perturbed'] = -1

    new_df_list = [orig_df, corr_df, incorr_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)


def perturb_delete(orig_df, row_index, perturb_change=True):
    new_df = deepcopy(orig_df)
    if perturb_change:
        new_df.loc[row_index:, 'is_perturbed'] = -1 if (new_df.iloc[row_index]['correct'] == 1) else 1
    new_df = new_df.iloc[:row_index].append(new_df.iloc[row_index+1:]).reset_index(drop=True)
    return new_df


def perturb_delete_random(orig_df, insert_policy=None):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'is_perturbed'] = 0
    orig_df.loc[:, 'orig_idx'] = orig_df.index
    if insert_policy == "first":
        row_idx = 0
    elif insert_policy == "middle":
        row_idx = len(orig_df) // 2
    elif insert_policy == "last":
        row_idx = len(orig_df) - 1
    else:
        row_idx = random.randrange(0, len(orig_df))
    new_df = perturb_delete(orig_df, row_idx)
    new_df.loc[:, 'user_id'] = new_df['user_id'].astype(str) + "_del"
    new_df_list = [orig_df, new_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)


def perturb_replace(orig_df, copy_idx, insert_idx, corr_value):
    del_df = perturb_delete(orig_df, insert_idx)
    new_df = perturb_insertion(del_df, copy_idx, insert_idx, corr_value)
    return new_df


def perturb_replace_random(orig_df, insert_policy=None):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'orig_idx'] = orig_df.index
    orig_df.loc[:, 'is_perturbed'] = 0
    copy_idx = random.randrange(0, len(orig_df)-1)
    if insert_policy == "first":
        insert_idx = 0
    elif insert_policy == "middle":
        insert_idx = len(orig_df) // 2
    elif insert_policy == "last":
        insert_idx = len(orig_df) - 1
    else:
        insert_idx = random.randrange(0, len(orig_df))
    corr = (orig_df.iloc[insert_idx]['correct'] == 0)
    if corr:
        new_df = perturb_replace(orig_df, copy_idx, insert_idx, 1)
        new_df.loc[:, 'user_id'] = new_df['user_id'].astype(str) + "_corr"
        new_df.loc[insert_idx + 1:, 'is_perturbed'] = 1
    else:
        new_df = perturb_replace(orig_df, copy_idx, insert_idx, 0)
        new_df.loc[:, 'user_id'] = new_df['user_id'].astype(str) + "_incorr"
        new_df.loc[insert_idx + 1:, 'is_perturbed'] = -1

    new_df_list = [orig_df, new_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)

