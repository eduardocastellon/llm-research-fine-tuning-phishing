import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
results_dir = base_dir / 'cleaned_results'

for model in ['mistral', 'llama', 'qwen']:
    csv_path = results_dir / f'{model}_results_cleaned.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing cleaned results file: {csv_path}')

    df = pd.read_csv(csv_path)
    # Find misclassified emails
    wrong = df[df['predicted_label'] != df['true_label']]
    # Show 3 examples where safe was predicted as phishing (false positive)
    fp = wrong[(wrong['true_label'] == 0) & (wrong['predicted_label'] == 1)]
    print(f"\n{model.upper()} false positives (safe → phishing):")
    print(fp['email_text'].head(3).tolist())