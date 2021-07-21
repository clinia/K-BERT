# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import pkuseg
import numpy as np
from transformers import AutoTokenizer
from rapidfuzz import fuzz, process

from utils.stop_words.stop_words import StopWords


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False, lang="chi"):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        if lang == "chi":
            self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        elif lang == "en":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

        # Import stop words filter
        self.stop_words = StopWords()

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        n_ent_found = 0
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))
                n_ent_found += len(entities)
                entities = entities[:max_entities]
                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    token_pos_idx = [pos_idx + i for i in range(1, len(token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(token) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), "constant")  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch, n_ent_found

    def add_knowledge_with_vm_en(
        self, sent_batch, labels_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128
    ):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        new_labels_batch = []
        n_ent_found = 0
        for sent, labels in zip(sent_batch, labels_batch):

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            new_labels = []
            max_iter = len(sent) - 1
            starts = ["B-", "I-"]
            start_idx = 0
            concatenated_entity = []
            for i in range(len(sent)):
                word = sent[i]
                next_word = sent[i + 1] if i != max_iter else ""
                to_lookup = word

                label = labels[i]

                # Concstenate possible entities
                concatenated_entity.append(word)

                # Discard possible entity if it starts with a stop word, else calculate the score
                if concatenated_entity[0] in self.stop_words.en:
                    score = 0
                    concatenated_entity = []
                else:
                    concatenated_entity_str = " ".join(concatenated_entity)
                    best_match, score, _ = process.extractOne(
                        concatenated_entity_str, self.lookup_table.keys(), scorer=fuzz.ratio
                    )

                # Keep concatenation only if score is above 60% threshold, else check score with next word
                if score < 60 or len(concatenated_entity) > 10:  # for safety
                    concatenated_entity = []
                else:
                    # Vheck if we have a better match with the next word
                    _, next_score, _ = process.extractOne(
                        concatenated_entity_str + " " + next_word, self.lookup_table.keys(), scorer=fuzz.ratio
                    )

                # Replace to looup only if score is greater than 95% threshold and next score is lower
                if score > 95 and next_score < score:
                    to_lookup = best_match
                    concatenated_entity = []

                # Safety for unwanted entities (probably due to a parsing error when building the lookup table)
                to_lookup = "" if to_lookup == "part" else to_lookup

                entities = list(self.lookup_table.get(to_lookup, []))
                n_ent_found += len(entities)
                entities = entities[:max_entities]
                tokens = self.tokenizer.tokenize(word)
                # Handle case where word cannot be tokenized (\xad, etc)
                if not tokens:
                    tokens = [word]

                # Adjust labels to to consider sub-word mapping
                new_labels.append(label)
                if len(tokens) > 1:
                    new_labels.extend(["O"] * (len(tokens) - 1))
                entity_tokens = [self.tokenizer.tokenize(entity) for entity in entities]
                sent_tree.append((tokens, entity_tokens))

                # if tokens in self.special_tags:
                #     token_pos_idx = [pos_idx + 1]
                #     token_abs_idx = [abs_idx + 1]
                # else:
                token_pos_idx = [pos_idx + i for i in range(1, len(tokens) + 1)]
                token_abs_idx = [abs_idx + i for i in range(1, len(tokens) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entity_tokens:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]

                know_sent += word
                seg += [0] * len(word)
                pos += pos_idx_tree[i][0]
                for j, ent in enumerate(sent_tree[i][1]):
                    know_sent.extend(ent)
                    seg += [1] * len(ent)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), "constant")  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
            new_labels_batch.append(new_labels)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch, new_labels_batch, n_ent_found
