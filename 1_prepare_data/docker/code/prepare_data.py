import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def to_sentiment(rating):
    rating = int(rating)
    # negative
    if rating <= 2:
        return 0
    # neutral
    elif rating == 3:
        return 1
    # positive
    else: 
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    input_folder = args.input
    output_folder = args.output
    seed = args.random_seed
    
    df = pd.read_csv(f'{input_folder}/reviews.csv')
    
    df['sentiment'] = df.score.apply(to_sentiment)
    
    train, test = train_test_split(df, test_size=0.1, random_state=seed)
    validation, test = train_test_split(test, test_size=0.5, random_state=seed)
    
    train.to_csv(f'{output_folder}/train.csv')
    validation.to_csv(f'{output_folder}/validation.csv')
    test.to_csv(f'{output_folder}/test.csv')