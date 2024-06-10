import os
import re
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Mapping

def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval()
    return config, tokenizer, model

def load_data(data_dir):
    train_data = None
    dev_data = None
    test_data = None
    for file in os.listdir(data_dir):
        data = pd.read_parquet(os.path.join(data_dir, file))
        if file.startswith("train"):
            train_data = data if train_data is None else pd.concat([train_data, data])
        elif file.startswith("valid") or file.startswith("dev"):
            dev_data = data if dev_data is None else pd.concat([dev_data, data])
        elif file.startswith("test"):
            test_data = data if test_data is None else pd.concat([test_data, data])
    return train_data, dev_data, test_data

def split_sentence(sentences, text):
    if text != "":
        return sentences + re.split(r"[\.\?\!\;\n]", text)
    else:
        return sentences

def get_sentences(data):
    sentences = reduce(split_sentence, [data.text for data in data.itertuples()], [])
    sentences = list(filter(lambda x: x.strip() != "", sentences))
    return sentences

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/wsy/models/gemma-2b")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/wsy/llama.cpp/early-stop/wikitext/wikitext-2-v1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="*cls**sent_0* It was *mask*.*sep*")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # Load the model and tokenizer
    config, tokenizer, model = load_model(args.model_name_or_path)
    model_name = config.model_type
    model.to(args.device)
    num_layers = -1
    if "num_hidden_layers" in config.__dict__:
        num_layers = config.num_hidden_layers
    print(f"Model: {model_name}, Number of layers: {num_layers}")

    # Load the data
    train_data, dev_data, test_data = load_data(args.data_dir)
    print(f"Train data size: {len(train_data)}, Dev data size: {len(dev_data)}, Test data size: {len(test_data)}")
    
    train_sents = get_sentences(train_data)
    print(f"Number of sentences in the train data: {len(train_sents)}")
    train_loader = torch.utils.data.DataLoader(train_sents, batch_size=args.batch_size, shuffle=False, drop_last=False)

    bar = tqdm(enumerate(train_loader))
    acc = 0
    for i, batch in bar:
        inputs = tokenizer(
            batch, 
            padding='max_length', 
            truncation=True if args.max_length > 0 else False, 
            max_length=args.max_length if args.max_length > 0 else None, 
            add_special_tokens=False,
            return_tensors="pt")
        inputs = {k: v.to(args.device) if type(v) == torch.Tensor else v for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[:-1]
        gold_logits = outputs.logits[:, -1].detach().cpu().numpy()
        gold_preds = gold_logits.argmax(axis=1)
        hidden_logits = [model.lm_head(model.model.norm(hidden_state)) for i, hidden_state in enumerate(hidden_states)]
        hidden_logits = [hidden_logit[:, -1].detach().cpu().numpy() for hidden_logit in hidden_logits]
        hidden_logits = np.stack(hidden_logits, axis=1)
        hidden_preds = hidden_logits.argmax(axis=2)
        acc += sum([1 if gold_pred in hidden_preds[i] else 0 for i, gold_pred in enumerate(gold_preds)])
        bar.set_description(f"Acc: {acc/np.min(((i+1)*args.batch_size, len(train_sents))):.2f}")
        del inputs, outputs
    print('Accuracy:', acc / len(train_sents))

if __name__ == "__main__":
    main()