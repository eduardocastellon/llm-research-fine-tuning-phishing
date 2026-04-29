import pandas as pd
import os
import sys

def clean_llm_results(df):
    def reparse(response):
        if not isinstance(response, str) or not response.strip():
            return -1
        response = response.lower().strip()
        words = response.split()
        if not words:
            return -1
        first_word = words[0]
        if 'phishing' in first_word:
            return 1
        if 'safe' in first_word:
            return 0
        for word in words:
            if 'phishing' in word:
                return 1
            if 'safe' in word:
                return 0
        return -1

    df['predicted_label'] = df['raw_response'].apply(reparse)
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python results_cleanup.py <path/to/file.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    
    print(f"\nProcessing {file_path}:")
    print(f"Total rows: {len(df)}")
    print(df['raw_response'].value_counts().head(10))
    
    df = clean_llm_results(df)
    
    print(f"\nAfter cleaning:")
    print(df['predicted_label'].value_counts())
    
    unrecognized = (df['predicted_label'] == -1).sum()
    print(f"Unrecognizable responses: {unrecognized} ({unrecognized/len(df)*100:.2f}%)")
    
    # Save cleaned file to ../cleaned_results/ with "_cleaned" suffix
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    cleaned_filename = f"{name}_cleaned{ext}"
    cleaned_dir = '../cleaned_results/'
    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(cleaned_dir, cleaned_filename)
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned file: {cleaned_path}")