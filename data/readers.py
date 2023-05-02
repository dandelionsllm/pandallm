import json
from typing import List
from data.data_utils import dfs_enumerate_all_assign

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class LSATReader:
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            for q in item['questions']:
                ques = q['question']
                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATProofWriterReader:
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            p_deductions = []
            iter_id = 0
            while f"prediction_{iter_id}" in item:
                p_deductions.append(item[f"prediction_{iter_id}"])
                iter_id += 1
            for q in item['questions']:
                ques = q['question']
                q_deductions = []
                iter_id = 0
                while f"prediction_{iter_id}" in q:
                    q_deductions.append(q[f"prediction_{iter_id}"])
                    iter_id += 1
                all_context.append(' '.join([passage] + p_deductions))
                all_question.append(' '.join([ques] + q_deductions))
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATReaderWPrompt:
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            for q in item['questions']:
                ques = q['question']
                deduction = q['prediction']
                all_context.append(passage + ' ' + deduction)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATReaderTrigger(LSATReader):
    relational_trigger_words = {
        'before': ['before', 'above', 'precede', 'earlier'],
        'after': ['after', 'larger', 'higher', 'bigger', 'older'],
        'last': ['immediately before', 'last'],
        'next': ['immediately after', 'next'],
        'adjacent': ['neighboring', 'adjacent'],
        'different': ['different'],
        'same': ['same', 'also'],
        'before_equal': ['no later'],
        'after_equal': ['no earlier'],
        'to': ['to', 'on', 'given', 'in']
    }
    relational_prompt = {
        'before': 'participant #1 is in the position before participant #2.',
        'after': 'participant #1 is in the position after participant #2.',
        'last': 'participant #1 is in the last position of participant #2.',
        'next': 'participant #1 is next to participant #2.',
        'adjacent': 'participant #1 is neighbouring to participant #2.',
        'different': 'participant #1 is in the different position with participant #2.',
        'same': 'participant #1 is in the same position with participant #2.',
        'before_equal': 'participant #1 is before or equals to the position of participant #2.',
        'after_equal': 'participant #1 is after or equals to the position of participant #2.',
        'to': 'participant #1 is assigned to the position #2.'
    }

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            for q in item['questions']:
                ques = q['question']

                prompts = []
                template = "The operation {} means that {}"
                for trigger, trigger_words in self.relational_trigger_words.items():
                    for _word in trigger_words:
                        if _word in ques or _word in passage:
                            prompts.append(template.format(_word, self.relational_prompt[trigger]))
                            break
                passage = ' '.join(prompts) + ' ' + passage

                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATReaderTriggerV2(LSATReader):
    relational_trigger_words = {
        'before': ['before', 'above', 'precede', 'earlier'],
        'after': ['after', 'larger', 'higher', 'bigger', 'older'],
        'last': ['immediately before', 'last'],
        'next': ['immediately after', 'next'],
        'adjacent': ['neighboring', 'adjacent'],
        'different': ['different'],
        'same': ['same', 'also'],
        'before_equal': ['no later'],
        'after_equal': ['no earlier'],
        'to': ['to', 'on', 'given', 'in']
    }
    relational_prompt = {
        # 'before': 'participant #1 is in the position before participant #2.',
        'before': 'If A does something {} B, then A is in the position before B.',
        # 'after': 'participant #1 is in the position after participant #2.',
        'after': 'If A does something {} B, then A is in the position after B.',
        # 'last': 'participant #1 is in the last position of participant #2.',
        'last': 'If A does something {} B, then A is in the last position of B.',
        # 'next': 'participant #1 is next to participant #2.',
        'next': 'If A does something {} B, then A is next to B.',
        # 'adjacent': 'participant #1 is neighbouring to participant #2.',
        'adjacent': 'If A does something {} to B, then A is neighbouring to B.',
        # 'different': 'participant #1 is in the different position with participant #2.',
        'different': 'If A does something {} to B, then A is in the different position with B.',
        # 'same': 'participant #1 is in the same position with participant #2.',
        'same': 'If A does something {} to B, then A is in the same position with B.',
        # 'before_equal': 'participant #1 is before or equals to the position of participant #2.',
        'before_equal': 'If A does something {} than B, then A is before or equals to the position of B.',
        # 'after_equal': 'participant #1 is after or equals to the position of participant #2.',
        'after_equal': 'If A does something {} than B, then A is after or equals to the position of B.',
        # 'to': 'participant #1 is assigned to the position #2.'
        'to': 'If A does something {} B, then A is assigned to the position of B.'
    }

    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        for item in data:
            passage = item['passage']
            for q in item['questions']:
                ques = q['question']

                prompts = []
                for trigger, trigger_words in self.relational_trigger_words.items():
                    for _word in trigger_words:
                        if _word in ques or _word in passage:
                            prompts.append(self.relational_prompt[trigger].format(_word))
                            break
                passage = ' '.join(prompts) + ' ' + passage

                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])

        return all_context, all_question, all_option_list, all_label


class LSATAssignmentEnumerationReader(LSATReader):
    def __call__(self, file: str):
        data = json.load(open(file, 'r'))
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        all_assignment = []

        for item in data:
            passage = item['passage']

            ent_group1 = item['group1']
            ent_group2 = item['group2']
            relation = item['relation']
            assignments = []
            dfs_enumerate_all_assign(ent_group1, ent_group2, relation, assignments, '', set(range(len(ent_group1))))
            item['all_assignment'] = assignments

            for q in item['questions']:
                ques = q['question']
                all_context.append(passage)
                all_question.append(ques)
                all_option_list.append(q['options'])
                all_label.append(self.label2id[q['answer']])
                all_assignment.append(assignments)

        return all_context, all_question, all_option_list, all_label, all_assignment


class LogicNLILangReader:
    label2id = {
        'contradiction': 0,
        'self_contradiction': 1,
        'neutral': 2,
        'entailment': 3
    }

    def __call__(self, file):
        data = json.load(open(file, 'r'))
        all_facts = []
        all_rules = []
        all_statements = []
        all_labels = []
        for item in data.values():
            fact_ls: List[str] = item['facts']
            rule_ls: List[str] = item['rules']
            for statement, label in zip(item['statements'], item['labels']):
                all_facts.append(fact_ls)
                all_rules.append(rule_ls)
                all_statements.append(statement)
                all_labels.append(self.label2id[label])

        return all_facts, all_rules, all_statements, all_labels


class ReClorReader:
    def __call__(self, file):
        data = json.load(open(file, 'r'))

        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        for sample in data:
            all_context.append(sample["context"])
            all_question.append(sample["question"])
            if "label" not in sample:
                all_label.append(-1)
            else:
                all_label.append(sample["label"])
            all_option_list.append(sample["answers"])

        return all_context, all_question, all_option_list, all_label


class ReClorLogicExpReader:
    base_name = "reclor_logic"

    def __init__(self, logic_expression_file, remove_underline: bool = False):
        logic_expressions = json.load(open(logic_expression_file))
        self.logic_expressions = {int(item[0]): item[1] for item in logic_expressions}
        self.remove_underline = remove_underline
        if remove_underline:
            self.base_name += "_r"

    def __call__(self, file):
        data = json.load(open(file, 'r'))
        cnt = 0

        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        for sample_id, sample in enumerate(data):
            context = sample["context"]
            question = sample["question"]
            answers = sample["answers"]
            if sample_id in self.logic_expressions:
                exp = self.logic_expressions[sample_id]
                if self.remove_underline:
                    exp = [item.replace("_", " ") for item in exp]

                context = context + "<logics>" + exp[0]
                answers = [ans + "<logics>" + exp[1 + ans_id] for ans_id, ans in enumerate(answers)]
                cnt += 1

            all_context.append(context)
            all_question.append(question)
            if "label" not in sample:
                all_label.append(-1)
            else:
                all_label.append(sample["label"])
            all_option_list.append(answers)

        logger.info(f"Enabled logical expressions samples: {cnt}")

        return all_context, all_question, all_option_list, all_label


class ReClorExampleReader:
    def __init__(self, retrieval_results: str, corpus_file: str, top_k: int):
        retrieval_results = json.load(open(retrieval_results))
        corpus = json.load(open(corpus_file))
        corpus = {item["id_string"]: f"{item['context']} {item['question']} {item['answers'][item['label']]}" for item in corpus}
        # self.examples = [
        #     [corpus[res[1]] for res in results[:top_k]] for results in retrieval_results
        # ]
        self.examples = {
            q: [corpus[res[1]] for res in v[:top_k]] for q, v in retrieval_results.items()
        }

    def __call__(self, file):
        data = json.load(open(file, 'r'))

        if len(data) != len(self.examples):
            logger.warning(f"Inconsistent data amount: {len(data)} v.s. {len(self.examples)}")

        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        for sample in data:
            examples = self.examples[sample["id_string"]]
            all_context.append("<s>".join(examples + [sample["context"]]))  # FIXME: Hard coding here.
            all_question.append(sample["question"])
            if "label" not in sample:
                all_label.append(-1)
            else:
                all_label.append(sample["label"])
            all_option_list.append(sample["answers"])

        return all_context, all_question, all_option_list, all_label


class LogiQAReader:
    label2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

    def __call__(self, file):
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        with open(file, 'r') as f:
            line = f.readline()
            idx = 0
            while line:
                idx += 1

                # blank line

                # right choice
                line = f.readline()
                all_label.append(self.label2id[line.strip()])
                idx += 1

                # context
                line = f.readline()
                all_context.append(line.strip())
                idx += 1

                # question
                line = f.readline()
                all_question.append(line.strip())
                idx += 1

                # options
                options = []
                for _ in range(4):
                    line = f.readline()
                    options.append(line.strip())
                all_option_list.append(options)

                line = f.readline()

        return all_context, all_question, all_option_list, all_label


class LogiQAReaderV2:
    def __call__(self, file):
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                all_label.append(item["answer"])
                all_context.append(item["text"])
                all_question.append(item["question"])
                all_option_list.append(item["options"])

        return all_context, all_question, all_option_list, all_label


class DreamReader:
    def __call__(self, file):
        data = json.load(open(file, 'r'))

        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        for dialogue in data:
            turns = dialogue[0]
            qas = dialogue[1]
            dial_id = dialogue[2]
            context = " ".join(turns)
            for qa in qas:
                all_context.append(context)
                all_question.append(qa["question"])
                all_option_list.append(qa["choice"])
                label = -1
                for c, choice in enumerate(qa["choice"]):
                    if choice == qa["answer"]:
                        label = c
                all_label.append(label)

        assert len(all_context) == len(all_question) == len(all_option_list) == len(all_label)
        return all_context, all_question, all_option_list, all_label
