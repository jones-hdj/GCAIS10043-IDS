import numpy as np
import pandas as pd
from tqdm import tqdm; tqdm.pandas()

# https://ocslab.hksecurity.net/Datasets/b-can-intrusion-dataset

def correct_label(row):
    # Attempts to label each part correctly, working around the strange formatting of this dataset
    try:
        dlc = int(row[2])  # Convert DLC from string to integer
        flag = row[2 + dlc]  # Locate the flag based on DLC
        row[2 + dlc] = np.nan  # Replace the original flag position with NaN
        row[11] = int(flag)  # Move the flag to column 11
    except (ValueError, IndexError):
        # Handle invalid DLC or out-of-bounds indexing
        # row[4] = np.nan  # Assign NaN to the flag column
        pass
    return row

# Read all of the individual datasets and apply the correct labels
df_norm = pd.read_csv('g80_bcan_normal_data.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_dos = pd.read_csv('g80_bcan_ddos_data.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_fuzz = pd.read_csv('g80_bcan_fuzzing_data.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)

# Shift the flag to the correct column due to strange formatting of the payload field(s)
df_norm[11] = df_norm[4].map(lambda x : 0)
df_norm.pop(4)
df_dos[11] = df_dos[4].map(lambda x : 1 if (x==1) else 0)
df_dos.pop(4)
df_fuzz[11] = df_fuzz[4].map(lambda x : 2 if (x==1) else 0)
df_fuzz.pop(4)

# Combine all of the datasets together
df_conc = pd.concat([df_norm,df_dos,df_fuzz])

# Filtering out any rows that contain the ErrorFrame CAN ID
df_conc = df_conc[~df_conc[2].str.contains('ErrorFrame', na=False)]

# Making sure all of the CAN IDs can be translated from hex to int
df_conc[1] = df_conc[1].apply(lambda x: x[-4:])
df_conc[1] = df_conc[1].apply(lambda x: '0x' + x if isinstance(x, str) and len(x) > 0 else x)

# Splitting up the payload field into individual bits
data_columns = df_conc[3].str.split(' ', expand=True)  # Split the data column by spaces
data_columns.columns = [i+3 for i in range(8)]  # Name the new columns: data_0, data_1, ..., data_7

# Adding in '0x' before each bit to allow for hex to int translation
for col in data_columns.columns:
    data_columns[col] = data_columns[col].apply(lambda x: '0x' + x if isinstance(x, str) and len(x) > 0 else x)

# Drop the original data column (column 1) and replace it with the new columns
df_conc = pd.concat([df_conc.drop(columns=3), data_columns], axis=1)

# Removing useless columns filled with NaNs
df_conc = df_conc.drop(columns=0)
df_conc = df_conc.dropna(axis=1, how='all')

# Sort the dataset columns by numerical value, ordering them correctly for the model
df_conc = df_conc[sorted(df_conc.columns)]

# Translate hex to int
df_conc[[1,3,4,5,6,7,8,9,10]] = df_conc[[1,3,4,5,6,7,8,9,10]].fillna('0').map(lambda x : int(x, 16))

# Open desired location for the full dataset and write to the .csv file
with open('bcan_dataset.csv', 'w') as f:
	for i,row in df_conc.iterrows():
		f.write(','.join([str(x) for x in row]) + '\n')
