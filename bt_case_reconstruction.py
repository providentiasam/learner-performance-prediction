import pandas as pd
import numpy as np
import random

# Deprecated
if 0:
    def gen_seq_reconstruction(
        data_df, 
        item_or_skill='item',
        min_sample_num=3, 
        min_thres=1,
        max_delay=np.inf,  #TODO
        ):
        user_key_sample_len = {}
        data_df['testpoint'] = np.nan
        keycol = '{}_id'.format(item_or_skill)
        for (user_id, key_id), user_key_df in data_df.groupby(['user_id', keycol]):
            if len(user_key_df) < min_sample_num:
                continue
            user_key_sample_len[(user_id, key_id)] = len(user_key_df)
            expand_win_avg = user_key_df['correct'].expanding(min_periods=min_sample_num).mean()
            test_points = expand_win_avg.loc[expand_win_avg.subtract(0.5).abs() >= abs(min_thres - 0.5)]
            if len (test_points):
                data_df.loc[test_points.index, 'testpoint'] = test_points.round()
    
        test_df_list = []
        new_user_id = data_df['user_id'].max() + 1
        for test_row in data_df.loc[data_df['testpoint'].notnull()].index:
            test_id = data_df[keycol][test_row]
            user_id = data_df['user_id'][test_row]
            user_df = data_df.loc[data_df['user_id'] == user_id]
            pre_df = user_df.loc[user_df.index <= test_row]
            post_df = user_df.loc[user_df.index > test_row]
            if test_id in post_df[keycol].unique():
                post_df = post_df.loc[:(post_df[keycol] == test_id).idxmax()].iloc[:-1]
            test_interaction = data_df.loc[[test_row]].copy()
            test_interaction['correct'] = test_interaction['testpoint']
            test_interaction['timestamp'] = np.nan
            # insert virtual test interaction into post_df
            post_df.reset_index(drop=True, inplace=True)
            post_df.index = post_df.index + 1
            insert_index = random.sample(range(post_df.shape[0] + 1), 1)[0]
            new_post_df = pd.concat(
                [post_df.loc[:insert_index],
                test_interaction], axis=0)
            new_df = pd.concat([
                pre_df.reset_index(drop=True),
                new_post_df.reset_index(drop=True)
                ], axis=0
            ).reset_index(drop=True)
            new_df['user_id'] = new_user_id
            new_df['timestamp'] = new_df['timestamp'].ffill()
            new_user_id += 1
            test_df_list.append(new_df)
        
        new_data = pd.concat(test_df_list, axis=0).reset_index(drop=True)
        data_meta = {
            'num_sample': new_data['user_id'].unique().shape[0],
            'num_interaction': new_data.shape[0],
        }
        return new_data, data_meta


def test_simple(bt_test_df, testcol='testpoint'):
    bt_test_df['testpass'] = (bt_test_df[testcol] == bt_test_df['model_pred'].round())
    groupby_key = ['all', testcol]
    return bt_test_df, groupby_key


def test_knowledge_state(bt_test_df, testcol='testpoint'):
    bt_test_df['testpass'] = (bt_test_df[testcol] - bt_test_df['model_pred']).abs()
    groupby_key = ['all', testcol]
    return bt_test_df, groupby_key


def gen_knowledge_state(
    data_df, 
    item_or_skill='item',
    common_test_set=None,
    max_test_per_user=100,
    max_seq_len=100,  
        # Use this to choose number of last interactions to use for testing.
        # This would make testing ground a bit more even for sakt / saint.
    ):
    data_df['testpoint'] = np.nan
    keycol = '{}_id'.format(item_or_skill)
    if item_or_skill == 'skill':
        raise NotImplementedError
    
    virtual_user_id = data_df['user_id'].max() + 1
    user_seqs = []
    user_item_meta = {}
    for user_id, user_df in data_df.groupby('user_id'):
        if max_seq_len is not None:  # Cut user sequence.
            user_df = user_df.tail(max_seq_len - 1)
        
        if common_test_set is None:  # Choose set of questions.
            user_test_set = pd.Series(user_df[keycol].unique()).iloc[:max_test_per_user].values
        else:
            user_test_set = common_test_set

        insert_ind = user_df.shape[0]
        for test_id in user_test_set:
            user_test_id_df = user_df.reset_index(drop=True)
            user_test_id_df = user_test_id_df.loc[user_test_id_df[keycol] == test_id]
            user_item_meta[(user_id, test_id)] = user_test_id_df
            new_seq = user_df.copy().reset_index(drop=True)
            new_seq.loc[insert_ind] = new_seq.iloc[-1]
            new_seq.loc[insert_ind, keycol] = test_id
            new_seq.loc[insert_ind, 'skill_id'] = user_test_id_df['skill_id'].iloc[0]
            new_seq.loc[insert_ind, 'testpoint'] = user_test_id_df['correct'].mean()
            new_seq.loc[insert_ind, 'correct'] = (user_test_id_df['correct'].mean() > 0.5).astype(float)
            new_seq['original_user_id'] = user_id
            new_seq['user_id'] = virtual_user_id
            virtual_user_id += 1
            user_seqs.append(new_seq)
    
    user_generated_df = pd.concat(user_seqs, axis=0).reset_index(drop=True)

    # TODO: Add additional data needed for
    return user_generated_df, user_item_meta
