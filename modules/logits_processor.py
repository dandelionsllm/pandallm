import torch
from transformers.generation_logits_process import LogitsProcessor

from modules.trie import Trie


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, trie: Trie, sent_mode: bool = False):
        self.trie = trie
        # If `sent_mode` is `True`, please ensure that each sentence in trie has two copies,
        # one ends with `<s>` and the other one ends with `<\s>` (eos token).
        self.sent_mode = sent_mode
        if sent_mode:
            assert self.trie.sep_token_id is not None
            self.sep_token_id = self.trie.sep_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sequence_ls = input_ids.tolist()
        scores_mask = scores.new_zeros(scores.size()).fill_(-10000.0)
        for seq_id, seq in enumerate(sequence_ls):
            if self.sent_mode:
                for idx in range(len(seq) - 1, -1, -1):
                    if seq[idx] == self.sep_token_id:
                        seq = seq[(idx + 1):]
            output = self.trie.get(seq)
            scores_mask[seq_id, output] = 0.0
        return scores + scores_mask
