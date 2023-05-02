from typing import Dict, List, Any

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin


class BLEUMetric(DistGatherMixin):
    def __init__(self):
        self.predictions = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        sources = []
        targets = []
        for item in meta_data:
            sources.append(item["src"])
            if "tgt" in item and item["tgt"]:
                targets.append(item["tgt"])
            else:
                targets.append("")

        pred_seq = batch_model_outputs["generated_seq"]
        predictions = [
            {
                "source": src,
                "target": tgt,
                "prediction": pred,
            } for src, tgt, pred in zip(sources, targets, pred_seq)
        ]

        if ddp:
            obj = predictions
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

        del meta_data, batch_model_outputs, sources, targets, pred_seq, predictions

    def get_results(self):
        bleu = sum(
            [sentence_bleu([word_tokenize(pred["target"])], word_tokenize(pred["prediction"])) for pred in
             self.predictions]
        ) * 1.0 / len(self.predictions)

        return {"bleu": bleu}, self.predictions
