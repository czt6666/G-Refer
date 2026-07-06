"""Self-contained BLEURT scorer using the official Google checkpoint directly
via its TensorFlow SavedModel signature (input_ids/input_mask/segment_ids ->
predictions), instead of installing the `bleurt` pip package (which isn't on
PyPI and is only available via `pip install git+https://github.com/google-
research/bleurt.git` -- installing/running an external repo's code, which
this environment's safety policy doesn't auto-approve).

Checkpoint: bleurt-base-128, downloaded directly from
https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip (Google's own
released weights -- data, not executable repo code) to
models/bleurt/bleurt-base-128/. Tokenization uses transformers' standard
BertTokenizer pointed at the checkpoint's own vocab.txt (already-installed,
general-purpose library code, not an external metric implementation).

Encoding matches BLEURT's convention: [CLS] reference [SEP] candidate [SEP],
padded/truncated to the checkpoint's max_seq_length (128 for bleurt-base-128).
"""
import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

_DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), os.pardir,
                             'models', 'bleurt', 'bleurt-base-128')


class BleurtScorer:
    def __init__(self, checkpoint=_DEFAULT_CKPT, max_seq_length=128):
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(checkpoint, 'vocab.txt'))
        self.model = tf.saved_model.load(checkpoint)
        self.sig = self.model.signatures['serving_default']

    def score(self, references, candidates, batch_size=32):
        assert len(references) == len(candidates)
        scores = []
        for i in range(0, len(references), batch_size):
            ref_batch = references[i:i + batch_size]
            cand_batch = candidates[i:i + batch_size]
            enc = self.tokenizer(ref_batch, cand_batch,
                                 padding='max_length', truncation=True,
                                 max_length=self.max_seq_length,
                                 return_tensors='tf')
            out = self.sig(input_ids=tf.cast(enc['input_ids'], tf.int64),
                           input_mask=tf.cast(enc['attention_mask'], tf.int64),
                           segment_ids=tf.cast(enc['token_type_ids'], tf.int64))
            scores.extend(out['predictions'].numpy().tolist())
        return scores


def bleurt_score(predictions, references, checkpoint=_DEFAULT_CKPT):
    """Matches eval_lite.py's calling convention: predictions=ground-truth
    chosen text, references=model generation (kept verbatim for comparability
    with the rest of this repo's eval scripts).
    """
    scorer = BleurtScorer(checkpoint)
    scores = scorer.score(references=predictions, candidates=references)
    return float(np.mean(scores)), float(np.std(scores))
