import os
import nltk
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def print_label_mapping(output_file):
    """Prints the class-to-label mapping and writes it to a file."""
    label_mapping = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech"
    }
    mapping_lines = ["Consists of class ids 1-4 where:"]
    for class_id, label_name in label_mapping.items():
        line = f"{class_id} -> {label_name}"
        mapping_lines.append(line)
        print(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mapping_lines))
    print(f"Class-to-label mapping saved to {output_file}")

def preprocess_text(text, stopwords):
    """Preprocesses text by removing punctuation, lowercasing, and removing stopwords."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

def preprocess_data(df, output_txt):
    """Preprocesses data and saves it in FastText format."""
    processed_data = []
    for _, row in df.iterrows():
        class_index = row['Class Index']
        title = row['Description']
        label = f"__label__{class_index}"
        cleaned_text = preprocess_text(title, stop_words)
        processed_data.append(f"{label} {cleaned_text}")
    with open(output_txt, 'w', encoding='utf-8') as f:
        for line in processed_data:
            f.write(line + '\n')
    print(f"Processed data saved to {output_txt}")

def prepare_datasets(train_csv, test_csv, output_dir):
    """Splits train.csv into train/dev, preprocesses all datasets, and saves them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print_label_mapping(os.path.join(output_dir, "class_mapping.txt"))

    df = pd.read_csv(train_csv)
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Class Index'])

    preprocess_data(train_df, os.path.join(output_dir, "train_fasttext.txt"))
    preprocess_data(dev_df, os.path.join(output_dir, "dev_fasttext.txt"))
    preprocess_data(pd.read_csv(test_csv), os.path.join(output_dir, "test_fasttext.txt"))