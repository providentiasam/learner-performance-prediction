import pandas as pd
import random
from tqdm import tqdm
import pickle

def gen_repeated_feed(
    data_df, 
    item_or_skill='item',
    repeat_val_list=[1, 0],
    repeat_length=50
    ):
    if item_or_skill == 'skill':
        raise NotImplementedError
    item2skill = data_df.groupby('item_id').first()['skill_id']
    df_list = []
    sorted_timestamps = data_df['timestamp'].sort_values()
    virtual_user_id = 0
    for item_id in tqdm(data_df[f'{item_or_skill}_id'].unique(), desc='Question-wise Test Construction'):
        for repeat_val in repeat_val_list:
            content_val_row = pd.Series({
                'user_id': virtual_user_id, 
                'item_id': item_id,
                'skill_id': item2skill[item_id],
                'correct': repeat_val,
            })
            content_val_df = pd.concat([content_val_row.to_frame().T \
                for _ in range(repeat_length)], axis=0).reset_index(drop=True)
            content_val_df['timestamp'] = sorted_timestamps.iloc[
                random.sample(list(range(len(sorted_timestamps))), repeat_length)].values
            virtual_user_id += 1
            df_list.append(content_val_df)
    total_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    total_df['testpoint'] = total_df['correct']
    return total_df, {}


def test_repeated_feed(bt_test_df):
    # bt_test_df['test_measure'] = (bt_test_df[testcol] == bt_test_df['model_pred'].round())
    groupby_key = ['all']
    return bt_test_df, groupby_key


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score, accuracy_score
    import matplotlib.pyplot as plt
    model = 'saint'
    dataset = 'ednet_small'
    testtype = 'repetition'

    config_name = f'{dataset}_{testtype}_{model}'
    with open(f'./results/{config_name}.pkl', 'rb') as file:
        bt_test_df = pickle.load(file)

    roc_auc_score(bt_test_df["correct"], bt_test_df['model_pred'])
    repeat_freq = 50
    bt_test_df['seq_ind'] = bt_test_df.index % repeat_freq
    bt_test_df.groupby(['seq_ind', 'correct'])['model_pred'].mean().unstack('correct').plot()
    plt.savefig(f'./{config_name}_avg.png')
    plt.close()
    bt_test_df.groupby(['seq_ind', 'correct'])['model_pred'].std().unstack('correct').plot()
    plt.savefig(f'./{config_name}_std.png')
    plt.close()

