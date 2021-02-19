import pandas as pd
import numpy as np
import random


def gen_continuity(data_df, num_sample_users=1000, max_seq_len=100, jump_step=5, num_items=50):
    # get N questions with max frequency in dataset
    item_set = get_question_set(data_df[['item_id', 'skill_id']], num_items)
    groupby_df = data_df.groupby('user_id')
    user_candidates = []
    for user_id, user_df in groupby_df:
        if len(user_df) >= max_seq_len:
            user_candidates.append((user_id, user_df))

    selected_users = random.sample(user_candidates, num_sample_users)
    virtual_user_id = 0
    user_seqs = []
    user_item_meta = {}

    for user_id, user_df in selected_users:
        user_df['original_user_id'] = user_df['user_id']
        user_df['cont_index'] = 0
        for interaction_idx in list(range(0, max_seq_len, jump_step)) + [max_seq_len]:
            sliced_seq = user_df.iloc[:interaction_idx]
            sliced_seq['cont_index'] = interaction_idx
            for item in item_set:
                new_row = user_df.iloc[0].copy() if interaction_idx == 0 else sliced_seq.iloc[-1].copy()
                new_row['item_id'] = item[0]
                new_row['skill_id'] = item[1]
                new_seq = pd.concat([sliced_seq.copy(), new_row], axis=0).reset_index(drop=True)
                new_seq['user_id'] = virtual_user_id
                virtual_user_id += 1
                user_seqs.append(new_seq)

    user_generated_df = pd.concat(user_seqs, axis=0).reset_index(drop=True)

    return user_generated_df, user_item_meta


def get_question_set(qids, num_questions):
    qids = qids.value_counts()[:num_questions].index.tolist()  # list of (item_id, skill_id)
    return qids
