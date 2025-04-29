import pandas as pd
from MCRNI import MCRNI


def main():
    # Load data
    df = pd.read_csv("input_data/CRT_sentiment.csv")

    y_true=df["label"]
    y_scores=df["GPT2_CNN_prob"]

    # TFIDF_RF_prob
    # BERT_CNN_prob
    # TFIDF_LR_prob
    # GPT2_CNN_prob

    # Initialize MCRNI model
    model = MCRNI(y_true, y_scores)

    # Compute MCRNI (against AUC threshold 0.5)
    model.compute_mcrni_with_auc(0.5)

    # Optionally print the evaluation report
    model.print_report()

    # Save the results to the results folder
    model.save_metrics(filename="CRT_sentiment_GPT2_CNN.csv", folder="results")

if __name__ == "__main__":
    main()
