import pandas as pd
import numpy as np
import random


def gen_continuity(data_df, num_sample_users=10, max_seq_len=100, jump_step=1, num_items=10):
    item2skill = data_df.groupby('item_id').first()['skill_id']
    # get N questions with max frequency in dataset
    item_set = get_question_set(data_df['item_id'], num_items)
    groupby_df = data_df.groupby('user_id')
    user_candidates = []
    for user_id, user_df in groupby_df:
        if len(user_df) >= max_seq_len:
            user_candidates.append((user_id, user_df))

    selected_users = random.sample(user_candidates, num_sample_users)
    virtual_user_id = 0
    user_seqs = []

    for user_id, user_df in selected_users:
        user_df['original_user_id'] = user_df['user_id']
        user_df['cont_index'] = 0
        user_df['is_virtual'] = 0
        for interaction_idx in list(range(0, max_seq_len, jump_step)) + [max_seq_len]:
            sliced_seq = user_df.iloc[:interaction_idx]
            sliced_seq['cont_index'] = interaction_idx
            for item_id in item_set:
                new_row = user_df.iloc[0].copy() if interaction_idx == 0 else sliced_seq.iloc[-1].copy()
                new_row['item_id'] = item_id
                new_row['skill_id'] = item2skill[item_id]
                new_row['correct'] = 1
                new_row['is_virtual'] = 1
                new_seq = pd.concat([sliced_seq.copy(), new_row.to_frame().T], axis=0).reset_index(drop=True)
                new_seq['user_id'] = virtual_user_id
                virtual_user_id += 1
                user_seqs.append(new_seq)

    user_generated_df = pd.concat(user_seqs, axis=0).reset_index(drop=True)
    other_info = {}
    return user_generated_df, other_info


def get_question_set(qids, num_questions):
    qids = qids.value_counts()[:num_questions].index.tolist()  # list of (item_id, skill_id)
    return qids


def test_continuity(result_df):
    virtual_only = result_df.loc[result_df['is_virtual']==1]
    org_user_dict = {}
    for org_user, user_df in virtual_only.groupby('original_user_id'):
        rfm_df = user_df.set_index(['item_id', 'cont_index'])['model_pred'].unstack('item_id')
        diff_df = rfm_df.diff()
        org_user_dict[org_user] = pd.concat([rfm_df, diff_df], axis=1, keys=['val', 'diff'])
    all_df = pd.concat([y for _, y in org_user_dict.items()], axis=0, keys=org_user_dict.keys())
    all_df = all_df.unstack(0)
    summary_dict = {
        'avg_abs_step_chg': all_df['diff'].abs().mean().mean(),
        'avg_final_chg': (all_df['val'].iloc[-1] - \
            all_df['val'].iloc[0]).abs().mean(),
        'avg_total_range': (all_df['val'].max(0) - all_df['val'].min(0)).mean(),
        'avg_max_step_chg': all_df['diff'].max(0).mean(),
        **({'step_chg_dist_' + x: y for x, y in \
            all_df['diff'].stack(0).stack(0).describe().to_dict().items()})
    }
    return result_df, pd.Series(summary_dict).to_frame().T


if __name__ == '__main__':
    pass
    # Score plot?