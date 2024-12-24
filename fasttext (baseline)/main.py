import fasttext
from preprocessing import prepare_datasets


def main():
    # Define file paths
    train_csv = "../data/train.csv"
    test_csv = "../data/test.csv"
    output_dir = "data"

    # Dataset paths
    train_data_path = f"{output_dir}/train_fasttext.txt"
    dev_data_path = f"{output_dir}/dev_fasttext.txt"
    test_data_path = f"{output_dir}/test_fasttext.txt"

    # Step 1: Data Preparation
    print("=" * 40)
    print("STEP 1: Data Preprocessing")
    print("=" * 40)
    prepare_datasets(train_csv, test_csv, output_dir)
    print("Data preprocessing completed.\n")

    # Step 2: Model Training
    print("=" * 40)
    print("STEP 2: Model Training")
    print("=" * 40)
    model = fasttext.train_supervised(
        input=train_data_path,
        autotuneValidationFile=dev_data_path,
        autotuneDuration=100,
        wordNgrams=2,
        verbose=1
    )
    print("Model training completed.\n")

    # Step 3: Model Evaluation
    print("=" * 40)
    print("STEP 3: Model Evaluation")
    print("=" * 40)
    result = model.test(test_data_path)

    # Extract metrics
    num_samples = result[0]
    precision = result[1]
    recall = result[2]
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print results
    print("Test Results:")
    print(f"Number of samples: {num_samples}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    main()
