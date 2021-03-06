import pandas as pd
import numpy as np
import random
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def gen_question_prior(
    data_df,
    item_or_skill='item',
    max_test_per_user=100
    ):
    if item_or_skill != 'item':
        raise NotImplementedError

    data_df['testpoint'] = np.nan
    keycol = f'{item_or_skill}_id'
    
    unique_item_df = data_df.groupby(keycol).first().reset_index()
    unique_item_df['correct'] = 1.0
    unique_item_df['testpoint'] = 1.0
    unique_item_df['user_id'] = unique_item_df[keycol] + data_df['user_id'].max() + 1

    item_occurrence = data_df.groupby(keycol).apply(len)
    item_correct_occurence = data_df.groupby(keycol)['correct'].apply(lambda x: x.astype(int).sum())
    item_meta = pd.concat([
        item_occurrence.to_frame('num_encounter'),
        item_correct_occurence.to_frame('num_correct')
    ], axis=1 
    )
    item_meta['prob_correct'] = item_meta['num_correct'] / item_meta['num_encounter']
    item_meta = item_meta.reset_index()

    return unique_item_df, item_meta


def test_question_prior(
    bt_test_df,
    item_meta,
    test_name='default'
    ):
    all_df = pd.concat([bt_test_df, item_meta], axis=1)

    all_df['test_measure_diff'] = all_df['model_pred'].subtract(all_df['prob_correct']).abs()
    all_df['test_measure'] = spearmanr(all_df['model_pred'].values, all_df['prob_correct'].values)[0]
    all_df['test_measure_corr'] = all_df[['model_pred', 'prob_correct']].corr().iloc[0,1]
    summary_dict = {
        'corr': all_df[['model_pred', 'prob_correct']].corr().iloc[0,1],
        'rankcorr': spearmanr(all_df['model_pred'].values, all_df['prob_correct'].values)[0],
        'probmae': all_df['model_pred'].subtract(all_df['prob_correct']).abs().mean()
    }
    summary_df = pd.Series(summary_dict).to_frame().T
    summary_df['groupby'] = 'all'
    summary_df['group'] = 'all'
    
    return all_df, summary_df


if __name__ == '__main__':
    import pickle
    question_prior_test_path = 'ednet_medium_question_prior_dkt'
    with open(f'./results/{question_prior_test_path}.pkl', 'rb') as file:
        all_df = pickle.load(file)
    all_df = pd.read_csv('./results/')
    all_df.plot(kind='scatter', x='prob_correct', y='model_pred')
    plt.savefig(f'./results/{question_prior_test_path}.png')
    plt.close()