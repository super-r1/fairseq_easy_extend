# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import numpy as np

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
from bleurt import score
import tqdm

@dataclass
class dummy(FairseqDataclass):
  pass

@register_scorer("bleurt", dataclass=dummy)
class BleurtScoreScorer(BaseScorer):
    def __init__(self, cfg):
        super(BleurtScoreScorer, self).__init__(cfg)
        self.cfg = cfg
        checkpoint = "/content/bleurt/bleurt/BLEURT-20"
        self.bleurt_score = score.BleurtScorer(checkpoint).score
        self.scores = None

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4):
        scores = np.zeros(len(self.pred))
        for index,(pred, target) in tqdm.tqdm(enumerate(zip(self.pred, self.ref))):
          scores[index] = self.bleurt_score(references=[target], candidates=[pred])[0]
        self.scores = scores
        return np.mean(self.scores)

    def result_string(self, order=4):
        return f"Bleurt: {self.score():.4f}"