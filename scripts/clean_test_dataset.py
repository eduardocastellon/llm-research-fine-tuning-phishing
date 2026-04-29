import pandas as pd

set1 = pd.read_csv("../data/raw/test/set1.csv")
set2 = pd.read_csv("../data/raw/test/set2.csv")


# Standardize set1
# 'body' -> 'email_text', label is already 0/1
set1_clean = pd.DataFrame({
    'email_text': set1['body'],
    'label': set1['label']  # 0 = safe, 1 = phishing
})

# Standardize set2
# 'Email Text' -> 'email_text', convert string labels to 0/1
set2_clean = pd.DataFrame({
    'email_text': set2['Email Text'],
    'label': set2['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
})

# Combine
df = pd.concat([set1_clean, set2_clean], ignore_index=True)

# Clean
df = df.drop_duplicates()
df = df.dropna()
df = df.reset_index(drop=True)

# check size
# print(df.shape)
# print(df['label'].value_counts())


# Check extremes
print(df[df['email_text'].str.len() < 50])        # too short
print(df[df['email_text'].str.len() > 10000].shape)  # too long

# Filter to reasonable range
df = df[df['email_text'].str.len() >= 50]
df = df[df['email_text'].str.len() <= 10000]
df = df.reset_index(drop=True)

print(df.shape)
print(df['email_text'].str.len().describe())
print(df['label'].value_counts())

# Save cleaned dataset to Drive
df.to_csv('../data/cleaned/test/clean_combined.csv', index=False)
print("Saved successfully")