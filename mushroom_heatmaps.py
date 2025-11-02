import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\holde\Downloads\mushrooms.csv", delimiter=',', header=0, na_values='?', keep_default_na=True)

data = data.dropna() # remove rows with '?' or any other NaN/NA values

class_mapping = {'e': True, 'p': False} # edible = True, poisonous = False

cols = { # column name: column data
    'cap shape': data['cap-shape'],
    'cap surface': data['cap-surface'],
    'cap color': data['cap-color'],
    'bruises': data['bruises'],
    'odor': data['odor'],
    'gill attachment': data['gill-attachment'],
    'gill spacing': data['gill-spacing'],
    'gill size': data['gill-size'],
    'gill color': data['gill-color'],
    'stalk shape': data['stalk-shape'],
    'stalk root': data['stalk-root'],
    'stalk surface above ring': data['stalk-surface-above-ring'],
    'stalk surface below ring': data['stalk-surface-below-ring'],
    'stalk color above ring': data['stalk-color-above-ring'],
    'stalk color below ring': data['stalk-color-below-ring'],
    'veil type': data['veil-type'],
    'veil color': data['veil-color'],
    'ring number': data['ring-number'],
    'ring type': data['ring-type'],
    'spore print color': data['spore-print-color'],
    'population': data['population'],
    'habitat': data['habitat']
}

# iterate through every column, create heatmap for each
for col, value in cols.items():
    matrix = pd.crosstab(value, data['class'])

    # Plot heatmap
    plt.figure(figsize=(8, max(4, len(matrix)*0.5)))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{col.title()} vs Class (Frequency Matrix)")
    plt.xlabel("Class")
    plt.ylabel(f"{col.title()}")
    plt.savefig(f'{col}_vs_class.png')
    plt.show()



