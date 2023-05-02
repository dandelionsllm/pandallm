import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def merge_data(data_dir, type):
    files = [f for f in os.listdir(f'{data_dir}/{type}') if f.endswith('.csv')]
    dfs = [pd.read_csv(f'{data_dir}/{type}/{f}', header=None) for f in files]
    df_all = pd.concat(dfs)

    df_new = pd.DataFrame(data={'inputs': [], 'targets': []})
    option_names = ['A', 'B', 'C', 'D']
    for i in tqdm(range(df_all.shape[0])):
        query, options = df_all.iloc[i, 0], df_all.iloc[i, 1:5].values
        options = [f'{name}. {opt}' for name, opt in zip(option_names, options)]
        query = [query] + options
        query = '\n'.join(query)
        answer = df_all.iloc[i, -1]
        row = pd.DataFrame({'inputs': [query], 'targets': [answer]})
        df_new = df_new.append(row, ignore_index=True)

    df_new.to_csv(f'{data_dir}/{type}.csv')
    print(f'{type} dataset merged successfully ...')



if __name__ == '__main__':
    data_dir = '/home/tianze/datasets/MMLU'

    for type in ['test', 'val', 'dev', 'auxiliary_train']:
        merge_data(data_dir, type)
