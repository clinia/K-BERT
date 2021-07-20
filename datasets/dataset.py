from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from ast import literal_eval
from uer.utils.constants import *
from multiprocessing import Pool, cpu_count


class NERInjectDataset(Dataset):
    def __init__(self, path, kg, vocab, labels_map, max_length, transform=None) -> None:
        super().__init__()

        self.path = path
        self.kg = kg
        self.vocab = vocab
        self.labels_map = labels_map
        self.max_length = max_length

        self.transform = transform
        self.dataset = pd.DataFrame()

        # Parallel processing
        n_cores = max(1, cpu_count() - 1)

        if self.path.endswith(".tsv"):

            with open(self.path, mode="r", encoding="utf-8") as f:
                f.readline()
                tokens, labels = [], []
                for line_id, line in enumerate(f):
                    tokens, labels = line.strip().split("\t")

                    text = "".join(tokens.split(" "))
                    tokens, pos, vm, tag = self.kg.add_knowledge_with_vm(
                        [text], add_pad=True, max_length=self.max_length
                    )
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")
                    tag = tag[0]

                    tokens = [self.vocab.get(t) for t in tokens]
                    labels = [self.labels_map[l] for l in labels.split(" ")]
                    mask = [1] * len(tokens)

                    new_labels = []
                    j = 0
                    for i in range(len(tokens)):
                        if tag[i] == 0 and tokens[i] != PAD_ID:
                            new_labels.append(labels[j])
                            j += 1
                        elif tag[i] == 1 and tokens[i] != PAD_ID:  # 是添加的实体
                            new_labels.append(labels_map["[ENT]"])
                        else:
                            new_labels.append(labels_map[PAD_TOKEN])

                    self.dataset = self.dataset.append(
                        {"tokens": tokens, "new_labels": new_labels, "mask": mask, "pos": pos, "vm": vm, "tag": tag},
                        ignore_index=True,
                    )

        elif self.path.endswith(".csv"):
            data = pd.read_csv(path, index_col=0, header=0, engine="python")
            data = data.applymap(lambda x: literal_eval(x))

            print("Processing dataset using {} cores.".format(n_cores))

            data_split = np.array_split(data, n_cores)
            pool = Pool(n_cores)
            self.dataset = pd.concat(pool.map(self.parallel_data_processing, data_split), ignore_index=True)
            pool.close()
            pool.join()

        print("Found {} entities".format(self.kg.n_ent_found))
        self.kg.n_ent_found = 0

    def __getitem__(self, index):
        sentence = self.dataset.loc[index]
        return (
            np.array(sentence["tokens"]),
            np.array(sentence["new_labels"]),
            np.array(sentence["mask"]),
            np.array(sentence["pos"]),
            np.array(sentence["vm"]),
            np.array(sentence["tag"]),
        )

    def __len__(self):
        return len(self.dataset)

    def parallel_data_processing(self, data):
        dataset = pd.DataFrame()
        data.reset_index(drop=True, inplace=True)
        for i in range(len(data)):
            text = data.loc[i, "text"]
            labels = data.loc[i, "tag"]

            tokens, pos, vm, tag, labels = self.kg.add_knowledge_with_vm_en(
                [text], [labels], add_pad=True, max_length=self.max_length
            )
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")
            tag = tag[0]
            labels = labels[0]

            tokens = [self.vocab.get(t) for t in tokens]
            labels = [self.labels_map[l] for l in labels]
            mask = [1] * len(tokens)

            new_labels = []
            j = 0
            for i in range(len(tokens)):
                if tag[i] == 0 and tokens[i] != PAD_ID:
                    new_labels.append(labels[j])
                    j += 1
                elif tag[i] == 1 and tokens[i] != PAD_ID:  # 是添加的实体
                    new_labels.append(self.labels_map["[ENT]"])
                else:
                    new_labels.append(self.labels_map[PAD_TOKEN])

            dataset = dataset.append(
                {"tokens": tokens, "new_labels": new_labels, "mask": mask, "pos": pos, "vm": vm, "tag": tag},
                ignore_index=True,
            )
        return dataset
