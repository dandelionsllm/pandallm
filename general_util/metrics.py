import collections
from typing import Dict, List, Any, Union, Tuple
import torch
from torch import distributed as dist
from transformers.models.t5.modeling_t5 import Seq2SeqModelOutput
from general_util.logger import get_child_logger
from general_util.mixin import DistGatherMixin

logger = get_child_logger("Metrics")

db_vocab = [
    "ccks_stock",
    "ccks_fund",
    "ccks_macro",
]


class DuSQLResultsHelper(DistGatherMixin):
    def __init__(self, db_schema_path: str):
        super(DuSQLResultsHelper, self).__init__()

        self.db_schema_path = db_schema_path

        self.predictions = []
        self.golds = []
        self.meta_data = []

    def __call__(self, meta_data: List[Dict[str, Any]],
                 batch_model_outputs: Union[Seq2SeqModelOutput, Dict[str, Any]],
                 ddp: bool = False):
        # Data format: qid\tsql_query\tdb_id

        pred_sql_query = batch_model_outputs["generated_seq"]
        golds = []
        predictions = []
        metas = []
        for item, pred in zip(meta_data, pred_sql_query):
            if "sql_query" in item:
                golds.append({
                    "q_id": item["q_id"],
                    "sql_query": item["sql_query"],
                    "db_id": item["db_name"]
                })
            predictions.append({
                "q_id": item["q_id"],
                "sql_query": pred,
                "db_id": item["db_name"]
            })
            metas.append({
                "parsing": item["parsing"],
                "question": item["question"],
                "db_name": item["db_name"],
            })

        if ddp:
            obj = list(zip(golds, predictions, metas))
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp_a, tmp_b, tmp_c = [], [], []
                for item in gather_res:
                    tmp_a.extend(list(map(lambda x: x[0], item)))
                    tmp_b.extend(list(map(lambda x: x[1], item)))
                    tmp_c.extend(list(map(lambda x: x[2], item)))
                golds = tmp_a
                predictions = tmp_b
                metas = tmp_c

        self.predictions.extend(predictions)
        self.golds.extend(golds)
        self.meta_data.extend(metas)

    def get_results(self):
        gold_dict = {
            item["q_id"]: [item["sql_query"], item["db_id"]] for item in self.golds
        }
        pred_dict = {
            item["q_id"]: [item["sql_query"], item["db_id"]] for item in self.predictions
        }

        # FIXME: Just as a example here and be replaced with any method calculating metrics.
        scores, _ = text2sql_evaluation.evaluate_complex_readin(self.db_schema_path, gold_dict, pred_dict, mode='exact', single_equal=True)

        logger.info(f"Full metrics: {scores}")

        for pred, meta in zip(self.predictions, self.meta_data):
            pred["question"] = meta["question"]
            pred["db_name"] = meta["db_name"]
            pred["parsing"] = meta["parsing"]

        return scores["all"], self.predictions


class RetrievalResultsBase:
    def __init__(self, top: Union[List[int], Tuple[int]] = (1, 3, 5, 10)):
        self.scores = []
        self.answer_ids: List[List[int]] = []
        self.top = top

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any]):
        self.scores.append(batch_model_outputs["scores"])
        self.answer_ids.extend([meta["answer_id"] for meta in meta_data])
        del batch_model_outputs, meta_data

    def get_results(self):
        scores = torch.cat(self.scores, dim=0)
        _, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        logger.info(sorted_indices.size())
        sorted_indices = sorted_indices.tolist()

        recall = collections.defaultdict(list)
        for answer_id, predictions in zip(self.answer_ids, sorted_indices):
            for k in self.top:
                recall_k = len(set(answer_id) & set(predictions[:k])) * 1.0 / len(answer_id)
                recall["recall@{}".format(k)].append(recall_k)

        res = {}
        for k in recall:
            res[k] = sum(recall[k]) * 1.0 / len(recall[k]) if len(recall[k]) > 0 else 0.0

        return res, sorted_indices
