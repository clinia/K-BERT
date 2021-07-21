# -*- encoding:utf -*-
"""
  This script provides an K-BERT example for NER.
"""
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import BertAdam
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
import numpy as np
import pandas as pd
from ast import literal_eval

from brain import KnowledgeGraph
from datasets.dataset import NERInjectDataset


class BertTagger(nn.Module):
    def __init__(self, args, model):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)
        # Target.
        output = self.output_layer(output)

        output = output.contiguous().view(-1, self.labels_num)
        output = self.softmax(output)

        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = (
            torch.zeros(label_mask.size(0), self.labels_num).to(torch.device(label.device)).scatter_(1, label, 1.0)
        )

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        predict = output.argmax(dim=-1)
        correct = torch.sum(label_mask * (predict.eq(label)).float())

        return loss, correct, predict, label


def main(special_args=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="Path of the pretrained model.")
    parser.add_argument(
        "--output_model_path", default="./models/tagger_model.bin", type=str, help="Path of the output model."
    )
    parser.add_argument(
        "--vocab_path", default="./models/google_vocab.txt", type=str, help="Path of the vocabulary file."
    )
    parser.add_argument("--train_path", type=str, default=None, help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, default=None, help="Path of the devset.")
    parser.add_argument("--test_path", type=str, default=None, help="Path of the testset.")
    parser.add_argument(
        "--config_path", default="./models/google_config.json", type=str, help="Path of the config file."
    )

    # Model options.
    parser.add_argument("--batch_size", type=int, default=16, help="Batch_size.")
    parser.add_argument("--seq_length", default=256, type=int, help="Sequence length.")
    parser.add_argument(
        "--encoder",
        choices=["bert", "lstm", "gru", "cnn", "gatedcnn", "attn", "rcnn", "crnn", "gpt", "bilstm"],
        default="bert",
        help="Encoder type.",
    )
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none", help="Subword feature type.")
    parser.add_argument(
        "--sub_vocab_path", type=str, default="models/sub_vocab.txt", help="Path of the subword vocabulary file."
    )
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg", help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100, help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # kg
    parser.add_argument("--kg_name", default=None, help="KG name or path")

    # tokenizer
    parser.add_argument("--tokenizer_config", default=None, help="Tokenizer config if english corpus")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path if english corpus")

    args = parser.parse_args()

    if special_args is not None:
        args.pretrained_model_path = special_args.pretrained_model_path
        args.config_path = special_args.config_path
        args.vocab_path = special_args.vocab_path
        args.train_path = special_args.train_path
        args.dev_path = special_args.dev_path
        args.test_path = special_args.test_path
        args.epochs_num = special_args.epochs_num
        args.batch_size = special_args.batch_size
        args.kg_name = special_args.kg_name
        args.output_model_path = special_args.output_model_path
        args.labels_path = special_args.labels_path

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    labels_map = {"[PAD]": 0, "[ENT]": 1}
    begin_ids = []

    # Find tagging labels & set language
    lang = None
    if args.train_path.endswith(".tsv"):
        lang = "chi"
        with open(args.train_path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                labels = line.strip().split("\t")[1].split()
                for l in labels:
                    if l not in labels_map:
                        if l.startswith("B") or l.startswith("S"):
                            begin_ids.append(len(labels_map))
                        labels_map[l] = len(labels_map)
    elif args.train_path.endswith(".csv"):
        lang = "en"
        labels = pd.read_csv(args.labels_path, index_col=0, names=["id", "label"], header=None).label.to_list()
        for label in labels:
            if label.startswith("B") or label.startswith("S"):
                begin_ids.append(len(labels_map))
            labels_map[label] = len(labels_map)

    print("Labels: ", labels_map)
    args.labels_num = len(labels_map)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    if args.kg_name == "none":
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=False, lang=lang)

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    # Build sequence labeling model.
    model = BertTagger(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Evaluation function.
    def evaluate(data_loader, args):

        instances_num = len(data_loader) * args.batch_size
        batch_size = args.batch_size

        print("Batch size: ", batch_size)
        print("The number of test instances:", instances_num)

        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0

        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)

        model.eval()

        for i, (
            input_ids_batch,
            label_ids_batch,
            mask_ids_batch,
            pos_ids_batch,
            vm_ids_batch,
            tag_ids_batch,
        ) in enumerate(data_loader):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)

            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    gold_entities_num += 1

            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    pred_entities_num += 1

            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    start = j
                    for k in range(j + 1, gold.size()[0]):

                        if gold[k].item() == labels_map["[ENT]"]:
                            continue

                        if (
                            gold[k].item() == labels_map["[PAD]"]
                            or gold[k].item() == labels_map["O"]
                            or gold[k].item() in begin_ids
                        ):
                            end = k - 1
                            break
                    else:
                        end = gold.size()[0] - 1
                    gold_entities_pos.append((start, end))

            for j in range(pred.size()[0]):
                if (
                    pred[j].item() in begin_ids
                    and gold[j].item() != labels_map["[PAD]"]
                    and gold[j].item() != labels_map["[ENT]"]
                ):
                    start = j
                    for k in range(j + 1, pred.size()[0]):

                        if gold[k].item() == labels_map["[ENT]"]:
                            continue

                        if (
                            pred[k].item() == labels_map["[PAD]"]
                            or pred[k].item() == labels_map["O"]
                            or pred[k].item() in begin_ids
                        ):
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    pred_entities_pos.append((start, end))

            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                else:
                    correct += 1

        print("Report precision, recall, and f1:")
        p = correct / pred_entities_num
        r = correct / gold_entities_num
        f1 = 2 * p * r / (p + r)
        print("{:.3f}, {:.3f}, {:.3f}".format(p, r, f1))

        return f1

    # Training phase.
    print("Building datasets.")
    # instances = read_dataset(args.train_path)

    # input_ids = torch.LongTensor([ins[0] for ins in instances])
    # label_ids = torch.LongTensor([ins[1] for ins in instances])
    # mask_ids = torch.LongTensor([ins[2] for ins in instances])
    # pos_ids = torch.LongTensor([ins[3] for ins in instances])
    # vm_ids = torch.BoolTensor([ins[4] for ins in instances])
    # tag_ids = torch.LongTensor([ins[5] for ins in instances])

    # Set up datasets
    train_dataset = NERInjectDataset(
        args.train_path, kg=kg, vocab=vocab, labels_map=labels_map, max_length=args.seq_length
    )
    dev_dataset = NERInjectDataset(args.dev_path, kg=kg, vocab=vocab, labels_map=labels_map, max_length=args.seq_length)
    # test_dataset = NERInjectDataset(
    #     args.test_path, kg=kg, vocab=vocab, labels_map=labels_map, max_length=args.seq_length
    # )

    instances_num = len(train_dataset)
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Start training.")
    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0},
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.0
    f1 = 0.0
    best_f1 = 0.0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (
            input_ids_batch,
            label_ids_batch,
            mask_ids_batch,
            pos_ids_batch,
            vm_ids_batch,
            tag_ids_batch,
        ) in enumerate(train_loader):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)

            loss, _, _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print(
                    "Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(
                        epoch, i + 1, total_loss / args.report_steps
                    )
                )
                total_loss = 0.0

            loss.backward()
            optimizer.step()

        # Evaluation phase.
        print("Start evaluate on dev dataset.")
        f1 = evaluate(args, False)
        print("Start evaluation on test dataset.")
        evaluate(args, True)

        if f1 > best_f1:
            best_f1 = f1
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    print("Final evaluation on test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    evaluate(args, True)


if __name__ == "__main__":
    main()
