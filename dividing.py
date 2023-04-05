import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file containing the classifications
df = pd.read_csv(r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\classifications.csv')

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['is_lens'])

# Save the train and test sets as CSV files
train_df.to_csv(r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\train.csv', index=False)
test_df.to_csv(r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\test.csv', index=False)