import pandas as pd

set1 = pd.read_csv("../data/raw/train/CEAS_08.csv")
set2 = pd.read_csv("../data/raw/train/Enron.csv")
set3 = pd.read_csv("../data/raw/train/Ling.csv")
set4 = pd.read_csv("../data/raw/train/Nazario.csv")
set5 = pd.read_csv("../data/raw/train/Nigerian_Fraud.csv")
set6 = pd.read_csv("../data/raw/train/phishing_email.csv")
set7 = pd.read_csv("../data/raw/train/SpamAssasin.csv")
set8 = pd.read_csv("../data/raw/train/The_Biggest_Spam_Ham_Phish_Email_Dataset.csv")


for name, df in [("CEAS_08", set1), ("Enron", set2), ("Ling", set3),
                  ("Nazario", set4), ("Nigerian_Fraud", set5),
                  ("phishing_email", set6), ("SpamAssasin", set7),
                  ("The_Biggest_Spam_Ham_Phish_Email_Dataset", set8)]:
    print(f"\n{name}:")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(df.head(2))



#combine all datasets

# CEAS_08 - body, label already 0/1
set1_clean = pd.DataFrame({
    'email_text': set1['body'],
    'label': set1['label']
})

# Enron - body, label already 0/1
set2_clean = pd.DataFrame({
    'email_text': set2['body'],
    'label': set2['label']
})

# Ling - body, label already 0/1
set3_clean = pd.DataFrame({
    'email_text': set3['body'],
    'label': set3['label']
})

# Nazario - body, label already 0/1
set4_clean = pd.DataFrame({
    'email_text': set4['body'],
    'label': set4['label']
})

# Nigerian_Fraud - body, label already 0/1
set5_clean = pd.DataFrame({
    'email_text': set5['body'],
    'label': set5['label']
})

# phishing_email - uses text_combined instead of body
set6_clean = pd.DataFrame({
    'email_text': set6['text_combined'],
    'label': set6['label']
})

# SpamAssasin - body, label already 0/1
set7_clean = pd.DataFrame({
    'email_text': set7['body'],
    'label': set7['label']
})

# The_Biggest - uses 'text', has 3 classes (0=ham, 1=phish, 2=spam)
# Drop spam (label == 2), keep only ham and phishing
set8_filtered = set8[set8['label'] != 2].copy()
set8_clean = pd.DataFrame({
    'email_text': set8_filtered['text'],
    'label': set8_filtered['label']  # 0=ham, 1=phishing already correct
})

print(f"The_Biggest after dropping spam: {set8_clean.shape}")
print(set8_clean['label'].value_counts())


# Combine all datasets
df = pd.concat([
    set1_clean, set2_clean, set3_clean,
    set4_clean, set5_clean, set6_clean,
    set7_clean, set8_clean
], ignore_index=True)

print(f"\nCombined dataset: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())


# Clean combined dataset

# Convert label to int to avoid type issues
df['label'] = df['label'].astype(int)

# Drop nulls and duplicates
df = df.dropna(subset=['email_text', 'label'])
df = df.drop_duplicates(subset=['email_text'])
df = df.reset_index(drop=True)

# Filter by length
df['email_text'] = df['email_text'].astype(str)
df = df[df['email_text'].str.len() >= 50]
df = df[df['email_text'].str.len() <= 10000]
df = df.reset_index(drop=True)

print(f"\nAfter cleaning: {df.shape}")
print(f"\nLabel distribution after cleaning:")
print(df['label'].value_counts())
print(f"\nNull values: {df.isnull().sum().sum()}")
print(f"Duplicate emails: {df.duplicated(subset=['email_text']).sum()}")


# Save cleaned training dataset
df.to_csv("../data/cleaned/train/train_combined.csv", index=False)
print(f"\nSaved to ../data/cleaned/train/train_combined.csv")
print(f"Final dataset size: {df.shape}")