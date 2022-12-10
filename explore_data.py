import matplotlib.pyplot as plt
import pandas as pd


def plot_df(df: pd.DataFrame, path_to_save: str, amount_of_labels: int = 3):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for col in df.columns[:-1]:
        print(col, end=' ')
        for k in range(amount_of_labels):
            plt.hist(df[df['label'] == k][col], color=colors[k], label='0', stacked=True, alpha=0.3, density=True, bins=15)
        plt.title(col)
        plt.ylabel('Probability')
        plt.xlabel(col)
        plt.legend()
        plt.savefig(path_to_save + 'orig\\' + col + '.png')


