import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

RESULTS_PATH = '../raw_results/'
CLEANED_RESULTS_PATH = '../cleaned_results/'

model_files = {
    'Mistral 7B':           (CLEANED_RESULTS_PATH, 'mistral_results_cleaned.csv'),
    'LLaMA 3.1 8B':         (CLEANED_RESULTS_PATH, 'llama_results_cleaned.csv'),
    'Qwen 2.5 7B':          (CLEANED_RESULTS_PATH, 'qwen_results_cleaned.csv'),
    'RoBERTa Baseline':     (RESULTS_PATH,          'roberta_baseline_results.csv'),
    'RoBERTa Fine-tuned':   (RESULTS_PATH,          'roberta_finetuned_results.csv')
}

print(f"{'Model':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Excluded':>10}")
print("=" * 75)

for model_name, (path, file) in model_files.items():
    full_path = f"{path}{file}"
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)

        # Count excluded
        excluded = (df['predicted_label'] == -1).sum()

        # Filter out -1
        df = df[df['predicted_label'] != -1]

        acc   = accuracy_score(df['true_label'], df['predicted_label'])
        f1    = f1_score(df['true_label'], df['predicted_label'])
        prec  = precision_score(df['true_label'], df['predicted_label'])
        rec   = recall_score(df['true_label'], df['predicted_label'])

        print(f"{model_name:<22} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f} {excluded:>10}")
    else:
        print(f"{model_name:<22} NOT FOUND at {full_path}")

print("\n" + "=" * 75)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 75)

for model_name, (path, file) in model_files.items():
    full_path = f"{path}{file}"
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df = df[df['predicted_label'] != -1]

        print(f"\n{model_name}:")
        print(classification_report(
            df['true_label'],
            df['predicted_label'],
            target_names=['Safe', 'Phishing']
        ))