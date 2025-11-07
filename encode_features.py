import pandas as pd

# open cleaned dataset
data = pd.read_csv(r"C:\Users\holde\mushroom_model\cleaned_mushroom.csv", 
                   delimiter=',', 
                   header=0, 
                   na_values='?', 
                   keep_default_na=True)

# choose features
columns = ["class",
           "odor", 
           "spore-print-color", 
           "ring-type", 
           "stalk-surface-below-ring", 
           "stalk-surface-above-ring"]

# one hot encode using chosen features, drop first, turn to 0 or 1
encoded = pd.get_dummies(data[columns], drop_first = True, dtype=int) 

# new csv with encoded values for chosen features
encoded.to_csv('cleaned_normalized_mushroom.csv', index=False)
