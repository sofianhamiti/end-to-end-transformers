import os
import argparse
import torch
from torch import nn
from utils import create_data_loader, train_model
from transformers import BertModel, BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # single gpu training


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, model_name):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default="")
    parser.add_argument("--batch_size", type=int, default="")
    parser.add_argument("--epochs", type=int, default="")
    parser.add_argument("--seed", type=int, default="")
    parser.add_argument("--max_len", type=int, default="")
    args, _ = parser.parse_known_args()
    

    model_name = args.model_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed
    max_len = args.max_len
    class_names = ['negative', 'neutral', 'positive']
    train_path = f'{args.data_folder}/train.csv'
    validation_path = f'{args.data_folder}/validation.csv'
    test_path = f'{args.data_folder}/test.csv'
    output_folder= args.output_folder
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # CREATE DATA LOADERS
    train_data_loader, df_train = create_data_loader(train_path, tokenizer, max_len, batch_size)
    val_data_loader, df_val = create_data_loader(validation_path, tokenizer, max_len, batch_size)
    
    # INSTANTIATE MODEL
    model = SentimentClassifier(len(class_names), model_name)
    model = model.to(device)
    
    train_model(model,
                train_data_loader,
                df_train,
                val_data_loader, 
                df_val,
                epochs,
                learning_rate,
                device,
                output_folder)
