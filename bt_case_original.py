import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

def test_original(result_df):
    summary_dict = {}
    result_df['user_idx'] = np.nan
    for uid, udf in tqdm(result_df.groupby('user_id'), desc='User-wise Item Indexing'):
        result_df.loc[udf.index, 'user_idx'] = range(udf.shape[0])
    result_df['mae'] = (result_df['correct'] - result_df['model_pred']).abs()

    df_dicts = {
        'all': result_df,
        'pos': result_df.loc[(result_df['correct'] == 1)],
        'neg': result_df.loc[(result_df['correct'] == 0)]}
    for tag, df_ in df_dicts.items():
        if tag == 'all':
            summary_dict[(tag, '', 'auc')] = roc_auc_score(df_['correct'], df_['model_pred'])
        summary_dict[(tag, '', 'acc')] = accuracy_score(df_['correct'], df_['model_pred'].round())
    summary_srs = pd.Series(summary_dict)
    summary_srs.index = pd.MultiIndex.from_tuples(list(summary_srs.index), names=['groupby', 'group', 'metric'])
    summary_df = summary_srs.unstack('metric')
    
    return result_df, summary_df


if __name__ == '__main__':
    pass


# def test_knowledge_state(bt_test_df, testcol='testpoint'):
#     bt_test_df['test_measure'] = (bt_test_df['avg'] - bt_test_df['model_pred']).abs()
#     groupby_key = ['all', testcol]
#     return bt_test_df, groupby_key


# def gen_knowledge_state(
#     data_df, 
#     item_or_skill='item',
#     common_test_set=None,
#     max_test_per_user=100,
#     max_seq_len=100,  
#         # Use this to choose number of last interactions to use for testing.
#         # This would make testing ground a bit more even for sakt / saint.
#     ):
#     data_df['testpoint'] = ''
#     data_df['avg'] = np.nan
#     keycol = '{}_id'.format(item_or_skill)
#     if item_or_skill == 'skill':
#         raise NotImplementedError
    
#     virtual_user_id = data_df['user_id'].max() + 1
#     user_seqs = []
#     user_item_meta = {}
#     for user_id, user_df in data_df.groupby('user_id'):
#         if max_seq_len is not None:  # Cut user sequence.
#             user_df = user_df.tail(max_seq_len - 1)
        
#         if common_test_set is None:  # Choose set of questions.
#             user_test_set = pd.Series(user_df[keycol].unique()).iloc[:max_test_per_user].values
#         else:
#             user_test_set = common_test_set

#         insert_ind = user_df.shape[0]
#         for test_id in user_test_set:
#             user_test_id_df = user_df.reset_index(drop=True)
#             user_test_id_df = user_test_id_df.loc[user_test_id_df[keycol] == test_id].copy()
#             user_item_meta[(user_id, test_id)] = user_test_id_df
#             new_seq = user_df.copy().reset_index(drop=True)
#             new_seq.loc[insert_ind] = new_seq.iloc[-1]
#             new_seq.loc[insert_ind, keycol] = test_id
#             new_seq.loc[insert_ind, 'skill_id'] = user_test_id_df['skill_id'].iloc[0]
#             new_seq.loc[insert_ind, 'avg'] = user_test_id_df['correct'].mean()
#             new_seq.loc[insert_ind, 'testpoint'] = ''.join(user_test_id_df['correct'].astype('str')\
#                 .replace('0', 'F').replace('1', 'T').values)
#             new_seq.loc[insert_ind, 'correct'] = (user_test_id_df['correct'].mean() > 0.5).astype(float)
#             new_seq['original_user_id'] = user_id
#             new_seq['user_id'] = virtual_user_id
#             virtual_user_id += 1
#             user_seqs.append(new_seq)
    
#     user_generated_df = pd.concat(user_seqs, axis=0).reset_index(drop=True)

#     # TODO: Add additional data needed for
#     return user_generated_df, user_item_meta