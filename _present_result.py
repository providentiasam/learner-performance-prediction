import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def summary2table(df):

    pass


if __name__ == '__main__':
    if 0:
        df_ms = pd.read_csv('./summary_ms.csv', index_col=0)
        df_yg = pd.read_csv('./summary_yg.csv', index_col=0)
        df_yg.index = pd.MultiIndex.from_tuples([eval(y) for y in df_yg.index], names=['group_by', 'group_tag'])
        df_yg = df_yg.reset_index(drop=False)
        df = pd.concat([df_ms, df_yg], axis=0).reset_index(drop=True)

        df['ratio'] = df['freq'] / df['count']
        df['acc'] = [(df['ratio'][i] if df['top'][i] else (1-df['ratio'][i])) for i in df.index]
        df['group_tag'] = [str(x).split('.')[0] for x in df['group_tag']]
        df['model'] = [x.upper() for x in df['model']]
        df.loc[:, 'group_tag'] = [{'0': 'Incorrect', '-1': 'Incorrect', '1': 'Correct', 'all': 'Any'}[x] for x in df['group_tag']]
        if 1:
            auc_table = df.set_index(['group_by', 'group_tag', 'test_type', 'dataset', 'model'])['auc'].dropna().xs('original', level='test_type')
            auc_table = auc_table.unstack('dataset')
            auc_table.index = auc_table.index.droplevel([0,1])
            auc_table.to_csv('./summary_auc.csv')


        df_acc = df.set_index(['group_by', 'group_tag', 'test_type', 'dataset', 'model'], append=True)['acc'].dropna()
        count_col = df.set_index(['group_by', 'group_tag', 'test_type', 'dataset', 'model'])['count'].replace(0, np.nan).dropna().unstack('dataset')
        
        df_acc.index = df_acc.index.droplevel(0)
        df_acc = df_acc.unstack('dataset').unstack('model').sort_index(axis=0, level=['test_type', 'group_tag'], ascending=[True, True]).dropna(how='all')
        # df_acc = df_acc.reset_index(drop=False)
        # df_acc.columns = \
        #     ['Assistments15', 'Assistments17', 'EdNet_Small', 'Spanish', 'Statics', 'TestCount']
        df_acc.to_csv('summary_formatted.csv')

        df['auc']