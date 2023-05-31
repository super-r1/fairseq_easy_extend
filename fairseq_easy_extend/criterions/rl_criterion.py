import math
from argparse import Namespace
import sys
sys.path.append("/content/bleurt")
from sys import builtin_module_names
from torchmetrics import BLEUScore, CHRFScore
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import encoders
from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field
from fairseq.logging import metrics
from bleurt import score
from comet import download_model, load_from_checkpoint
import sacrebleu
import nltk
from nltk.translate import meteor_score
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metrics bleu sacrebleu chrf meteor bleurt or comet"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tokenizer = encoders.build_tokenizer(Namespace(
            tokenizer='moses'
        ))
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.bleu_score = BLEUScore(n_gram=4)
        self.chrf_score = CHRFScore()
        self.meteor_score = meteor_score.single_meteor_score
        self.sacrebleu_score = sacrebleu.corpus_bleu
        if self.metric == "comet":
          model_path = download_model("Unbabel/wmt22-comet-da")
          comet_model = load_from_checkpoint(model_path)
          self.comet_score = comet_model.predict
        elif self.metric == "bleurt":
          checkpoint = "/content/bleurt/bleurt/BLEURT-20"
          self.bleurt_score = score.BleurtScorer(checkpoint).score
            
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        #get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss, reward = self._compute_loss(outs, tgt_tokens, src_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward.detach()
        }

        return loss, sample_size, logging_output

    def decode(self, toks, escape_unk=False):
        with torch.no_grad():
            s = self.tgt_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            s = self.tokenizer.decode(s)
        return s
    def decode_source(self, toks, escape_unk=False):
        with torch.no_grad():
            s = self.src_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            s = self.tokenizer.decode(s)
        return s
    def compute_reward(self, outputs, targets, src_tokens):
        """
        #we take a softmax over outputs
        probs = F.softmax(outputs, dim=-1)
        #argmax over the softmax \ sampling (e.g. multinomial)
        samples_idx = torch.multinomial(probs, 1, replacement=True)
        sample_strings = self.tgt_dict.string(samples_idx)  #see dictionary class of fairseq
        #sample_strings = "I am a sentence"
        reward_vals = evaluate(sample_strings, targets)
        return reward_vals, samples_idx
        """
        # bsz = outputs.size(0)
        # seq_len = outputs.size(1)
        # vocab_size = outputs.size(2)

        # probs = F.softmax(outputs, dim=-1).view(bsz * seq_len, vocab_size)
        # sample_idxs = torch.argmax(probs, dim=1).view(bsz, seq_len)
        # rewards = torch.zeros(bsz, device='cuda')

        # for batch_idx in range(bsz):
        #     batch_mask = masks[batch_idx]
        #     sample_idxs_masked = sample_idxs[batch_idx] * batch_mask
        #     targets_masked = targets[batch_idx] * batch_mask
        #     inputs = src_tokens[batch_idx]
        #     sampled_sentence_string = self.decode(sample_idxs_masked.unsqueeze(0))
        #     target_sentence = self.decode(targets_masked.unsqueeze(0))
        #     inputs  = self.decode_source(inputs)
        #     reward = self._evaluate(sampled_sentence_string, target_sentence, inputs)
        #     rewards[batch_idx] = reward
        # return reward_vals, samples_idx
        pass

    def _evaluate(self, sampled_string, target, source):
      
      if self.metric == "bleu":
        reward = self.bleu_score([sampled_string], [[target]]) 
      elif self.metric == "chrf": 
        reward = self.chrf_score([sampled_string], [[target]])
      elif self.metric == "meteor":
        reward = self.meteor_score(sampled_string.split(" "), target.split(" "))
      elif self.metric == "sacrebleu":
        reward = self.sacrebleu_score([sampled_string], [[target]]).score
      elif self.metric == "bleurt":
        reward = self.bleurt_score(references=[target], candidates=[sampled_string])[0]
      elif self.metric == "comet":
        data = [{
          "src": source,
          "mt": sampled_string,
          "ref": target
        }]
        reward = self.comet_score(data, batch_size=8, gpus=1).system_score
      else:
        raise Exception(f"I do not know what is {self.metric} I am aware of only bleu bleurt and comet")
      return reward

    def _compute_loss(self, outputs, targets, src_tokens, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """
        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)

        # Get the probabilities over vocab size
        probs = F.softmax(outputs, dim=-1).view(bsz * seq_len, vocab_size)
        sample_idxs = torch.argmax(probs, dim=1).view(bsz, seq_len)
        rewards = torch.zeros(bsz, device='cuda')

        for batch_idx in range(bsz):
            batch_mask = masks[batch_idx]
            sample_idxs_masked = sample_idxs[batch_idx] * batch_mask
            targets_masked = targets[batch_idx] * batch_mask
            inputs = src_tokens[batch_idx]
            sampled_sentence_string = self.decode(sample_idxs_masked.unsqueeze(0))
            target_sentence = self.decode(targets_masked.unsqueeze(0))
            inputs  = self.decode_source(inputs)
            reward = self._evaluate(sampled_sentence_string, target_sentence, inputs)
            rewards[batch_idx] = reward

        reward = rewards.unsqueeze(1).repeat(1, seq_len)
        reward.requires_grad = True

        if masks is not None:
          outputs, targets = outputs[masks], targets[masks]
          reward, sample_idx = reward[masks], sample_idxs[masks]

        # Log probabiltity of evereything
        log_probs = torch.log(probs)
        # Gather the part we sampled.
        log_probs_of_samples = torch.gather(log_probs, 1, sample_idx.unsqueeze(1))
        loss = -log_probs_of_samples * reward.unsqueeze(1)
        loss = loss.mean()
        # For more about mask see notes on NLP2-notes-on-mask
        return loss, reward.mean()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        reward_sum= sum(log.get("reward", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("reward", reward_sum / sample_size, sample_size, round=3)
