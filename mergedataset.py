import pandas as pd

# Load the first dataset
df1 = pd.read_csv("./Preprocessing_dataset/namified_Ben0_with_Trojan.csv")

# Load the second dataset
df2 = pd.read_csv("./Preprocessing_dataset/namified_Torjan_with_Trojan.csv")

# Merge the datasets
import pandas as pd

# Assuming you have two dataframes df1 and df2
# Merge them row-wise
merged_df = pd.concat([df1, df2])

# Reset index if necessary
merged_df.reset_index(drop=True, inplace=True)

# Display the merged dataframe
print(merged_df)


merged_df.to_csv('merged_dataset.csv', index=False)

