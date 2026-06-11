# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Reconstructed for G-Refer reproduction.
# The upstream G-Refer repo omitted ds_training/utils/data/. This file restores
# the DeepSpeed-Chat raw-dataset abstraction plus the `local/jsonfile` dataset
# that G-Refer's RAFT step (run_{dataset}.sh, --data_path local/jsonfile--<name>)
# relies on. Records in raft_data/<name>/{train,eval,test}.json have the fields
# {"uid", "iid", "prompt", "chosen", "reject"}.
from datasets import load_dataset


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


class LocalJsonFileDataset(PromptRawDataset):
    """Reads raft_data/<name>/{train,eval}.json.

    Each json line/record carries: uid, iid, prompt, chosen, reject.
    For G-Refer the prompt already contains the full instruction and ends with
    '### Explanation:', and `chosen` is the gold explanation. We therefore pass
    the fields through verbatim (no Human/Assistant wrapping) so the training
    text matches the format used at inference time (ds_inference/infer.py).
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        # Load train/eval with separate calls: G-Refer's train.json carries an
        # extra `similarity_score` column that eval.json lacks, which makes a
        # single combined load_dataset call fail on the column-set mismatch.
        self.raw_datasets = {
            "train":
            load_dataset('json',
                         data_files={"train": chat_path + '/train.json'},
                         split="train"),
            "eval":
            load_dataset('json',
                         data_files={"eval": chat_path + '/eval.json'},
                         split="eval"),
        }

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return sample['prompt']
        return None

    def get_chosen(self, sample):
        if sample['chosen'] is not None:
            return sample['chosen']
        return None

    def get_rejected(self, sample):
        # G-Refer uses the field name "reject"; fall back to "rejected".
        val = sample.get('reject', sample.get('rejected', None))
        if val is not None:
            return val
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['chosen'] is not None:
            return sample['prompt'] + sample['chosen']
        return None

    def get_prompt_and_rejected(self, sample):
        rejected = self.get_rejected(sample)
        if sample['prompt'] is not None and rejected is not None:
            return sample['prompt'] + rejected
        return None
