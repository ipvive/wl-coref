""" Describes AttenEncoder. Extracts attention from bert-encoded text.
"""

from typing import Tuple

import torch

from coref.config import Config
from coref.const import Doc

import pdb

class AttnEncoder(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """ Receives bert attention of a text, extracts all the
    possible mentions in that text. """

    def __init__(self, dropout_rate,
                       num_heads,
                       num_layers):
        """
        Args:
            features (int): the number of featues in the input embeddings
            config (Config): the configuration of the current session
        """
        super().__init__()
        self.coref = torch.nn.Linear(in_features=num_layers * num_heads, out_features=1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.coref.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc: Doc,
                attns: torch.Tensor
                ) -> Tuple[torch.Tensor, ...]:
        """
        Extracts word representations from text.

        Args:
            doc: the document data
            x: a tensor containing bert output, shape (n_subtokens, bert_dim)

        Returns:
            words: a Tensor of shape [n_words, mention_emb];
                mention representations
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-coreferent words have cluster id of zero.
        """
        word_boundaries = torch.tensor(doc["word2subword"], device=self.device)
        starts = word_boundaries[:, 0]
        ends = word_boundaries[:, 1]

        s = attns.shape
        attns = torch.reshape(attns, (s[0], s[1]*s[2], s[3], s[4]))
        attns = torch.transpose(attns, 1, 3)
        attns = torch.transpose(attns, 1, 2)
        
        subword_coref_attn = torch.squeeze(self.coref(attns), dim=-1)

        sub_to_word = self._subword_to_word(starts, ends, s[-1])

        word_coref_attn = torch.matmul(sub_to_word, subword_coref_attn)
        word_coref_attn = torch.matmul(word_coref_attn, sub_to_word.t())

        # we only have one batch, so we debatch here
        return word_coref_attn[0]

    def _subword_to_word(self,
                     word_starts: torch.Tensor,
                     word_ends: torch.Tensor,
                     n_subtokens) -> torch.Tensor:
        """ 
        Args:
            word_starts (torch.Tensor): [n_words], start indices of words
            word_ends (torch.Tensor): [n_words], end indices of words

        Returns:
            torch.Tensor: [description]
        """
        n_words = len(word_starts)

        # [n_mentions, n_subtokens]
        # with 0 at positions belonging to the words and -inf elsewhere
        sub_to_word = torch.arange(
            0, n_subtokens, device=self.device).expand((n_words, n_subtokens))
        sub_to_word = ((sub_to_word >= word_starts.unsqueeze(1))
                     * (sub_to_word < word_ends.unsqueeze(1))).to(torch.float)

        s = torch.sum(sub_to_word, dim=1)

        return sub_to_word / torch.reshape(s, (-1,1)) 
