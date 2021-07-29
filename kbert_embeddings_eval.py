import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from uer.utils.seed import set_seed
from uer.utils.config import load_hyperparam
from uer.model_builder import build_model
from uer.utils.vocab import Vocab
from main_ner_evaluate import NERKBERTArgs

from search_utils.embeddings_eval.base import EvaluateEmbeddings


class EvalKBERTEMbeddings(EvaluateEmbeddings):
    def _forward(self, model, inputs, labels, tokenizer, device):

        ent_emb = []
        for i in range(2):
            out = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=list(inputs[i]),
                add_special_tokens=True,
                max_length=self.seq_len,
                padding=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_overflowing_tokens=False,
                return_special_tokens_mask=False,
            )

            input_ids = torch.LongTensor(out["input_ids"]).to(device)
            attention_mask = torch.LongTensor(out["attention_mask"]).to(device)
            token_type_ids = torch.LongTensor(out["token_type_ids"]).to(device)
            token_type_ids = torch.ones_like(token_type_ids).to(
                device
            )  # They use ones as their ids for single sequences

            # create pos embedding according to how the authors of K-BERT did it
            batch_size, seq_len = input_ids.shape
            pos = torch.arange(start=0, end=seq_len).repeat(batch_size, 1)
            pos = torch.mul(pos, attention_mask)
            temp_mask = torch.zeros_like(pos)
            temp_mask[pos == 0] = 1
            temp_mask[:, 0] = 0  # do not apply correction to first positions
            temp_mask = temp_mask * (seq_len - 1)
            pos = (pos + temp_mask).type(torch.long).to(device)

            # forward pass to the model
            # output = model(input_ids, attention_mask, token_type_ids, pos)
            wd_emb = model.embedding(input_ids, token_type_ids, pos)
            ctx_emb = model.encoder(wd_emb, attention_mask)  # Change model.encoder to get more hidden states!

            ## Separate embeddings
            # Remove special token indexes from token_tyoe_ids
            # first segment
            token_type_ids = torch.where(
                ((input_ids == 101) | (input_ids == 102) | (input_ids == 0)), 0, token_type_ids
            )

            ## Select embeddings
            emb1 = ctx_emb[token_type_ids == 1, :]
            n_sub_tokens = torch.sum(token_type_ids, dim=1)

            ent_emb_i = []
            for n in n_sub_tokens:
                tmp = emb1[: int(n)].sum(0)
                emb1 = emb1[int(n) :]
                ent_emb_i.append(tmp)

            ent_emb.append(torch.stack(ent_emb_i))

        return ent_emb[0], ent_emb[1]


def main(DATA_PATH, EXPORT_PATH, special_args=None):

    ####################################
    # Same soup to call the model

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
        args.seq_length = special_args.seq_length

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Define device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize with pretrained model.
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device(device)), strict=False)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    ##########################
    # Feed loaded model and tokenizer to the embedding evaluation module

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    evaluator = EvalKBERTEMbeddings(DATA_PATH=DATA_PATH, batch_size=10, seq_len=30)
    results = evaluator.evaluate(model, tokenizer)

    evaluator.export(results=results, EXPORT_PATH=EXPORT_PATH)


if __name__ == "__main__":
    DATA_PATH = "./datasets/embedding_eval/data.csv"
    EXPORT_PATH = "./outputs/embedding_eval"

    # Load model-related args
    args = NERKBERTArgs(model=2)

    main(DATA_PATH, EXPORT_PATH, args)
