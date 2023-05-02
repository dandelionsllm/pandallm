import copy
from typing import List, Set, Union, Dict, Tuple

from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast, AlbertTokenizer, AlbertTokenizerFast, DebertaTokenizer, \
    DebertaTokenizerFast, DebertaV2Tokenizer
from transformers.models.bert.tokenization_bert import whitespace_tokenize

from general_util.logger import get_child_logger

try:
    from nltk import word_tokenize
except:
    pass

logger = get_child_logger(__name__)


def tokenizer_get_name(_tokenizer: PreTrainedTokenizer):
    tokenizer_name = _tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
    return tokenizer_name


def get_sep_tokens(_tokenizer: PreTrainedTokenizer):
    if _tokenizer.sep_token:
        return [_tokenizer.sep_token] * (_tokenizer.max_len_single_sentence - _tokenizer.max_len_sentences_pair)
    return []


# FIXED: This method may find a span within a single word.
# def find_span(text: str, span: str, start: int = 0):
#     pos = text.find(span, start)
#     if pos == -1:
#         return []
#     _e = pos + len(span)
#     return [(pos, _e)] + find_span(text, span, start=_e)


def is_alphabet(char):
    res = ord('a') <= ord(char) <= ord('z') or ord('A') <= ord(char) <= ord('Z')
    res = res or (char in ['-', '\''])  # Fix the problem shown in the bad case of the method `span_chunk`.
    return res


def whitespace_tokenize_w_punctuation_ends(text):
    """
    Bad case:
    >>> whitespace_tokenize_w_punctuation_ends("\" My name is Fangkai Jiao.\"")
    >>> ['"', 'My', 'name', 'is', 'Fangkai', 'Jiao.', '"']

    >>> word_tokenize("\" My name is Fangkai Jiao.\"")
    >>> ['``', 'My', 'name', 'is', 'Fangkai', 'Jiao', '.', "''"]
    """
    words = whitespace_tokenize(text)
    new_words = []
    for word in words:
        if len(word) == 1:
            new_words.append(word)
            continue

        if not is_alphabet(word[0]):
            new_words.append(word[0])
            word = word[1:]

        if len(word) == 1:
            new_words.append(word)
            continue

        if not is_alphabet(word[-1]):
            new_words.append(word[:-1])
            new_words.append(word[-1])
        else:
            new_words.append(word)

    return new_words


def find_span(sentence: str, span: str, start: int = 0):
    span = span.strip()

    s = sentence.find(span, start)
    if s == -1:
        return []

    e = s + len(span)

    a = not is_alphabet(sentence[s - 1]) if s > 0 else True
    b = not is_alphabet(sentence[e]) if e < len(sentence) else True
    if a and b:
        return [(s, e)] + find_span(sentence, span, start=e)
    else:
        return find_span(sentence, span, start=e)


def span_chunk(text: str, span_ls: List[str], space_tokenize: bool = False) -> Tuple[List[str], List[int]]:
    """
    Word based span indicating.
    The method is based on whitespace tokenization, which may lead to inconsistent with BPE or Wordpiece.

    FIXME:
        1. The warnings are to be fixed. There is some consistency can be address through proper text normalization.
        2. The `whitespace_tokenize` aims to not split the words such as "don't",
            but may cause the punctuations not split correctly.
    """
    pos_ls = []
    for span in span_ls:
        span_pos_ls = find_span(text, span)
        pos_ls.extend(span_pos_ls)
    pos_ls = sorted(pos_ls, key=lambda x: x[0])

    # Unified span
    to_be_dropped = set()
    for i, pos_i in enumerate(pos_ls):
        for j, pos_j in enumerate(pos_ls):
            if i == j:
                continue
            if pos_j[0] <= pos_i[0] and pos_i[1] <= pos_j[1]:
                to_be_dropped.add(i)

    new_pos_ls = []
    for pos_id, pos in enumerate(pos_ls):
        if pos_id not in to_be_dropped:
            new_pos_ls.append(pos)
    pos_ls = new_pos_ls

    # No within word span check.
    for pos_id, pos in enumerate(pos_ls):
        if pos_id == 0:
            continue
        # assert pos[0] >= pos_ls[pos_id - 1][1], (span_ls, text[pos[0]: pos[1]], text[pos_ls[pos_id - 1][0]: pos_ls[pos_id - 1][1]])
        # TODO: Think about how to fix this:
        #   some bad cases:
        #   - AssertionError: (['Goethe-Universität', 'Johann Wolfgang Goethe'], 'Goethe-Universität', 'Johann Wolfgang Goethe')
        if pos[0] < pos_ls[pos_id - 1][1]:
            # pos[0] = pos_ls[pos_id - 1][1]
            pos_ls[pos_id] = (pos_ls[pos_id - 1][1], pos[1])

    text_spans = []
    indicate_mask = []
    last_e = 0
    for s, e in pos_ls:
        if last_e > s:
            logger.warning(f"Overlapped span: {text_spans[-1]}\t{text[s: e]}\t{text}")
            print(f"Overlapped span: {text_spans[-1]}\t{text[s: e]}\t{text}")
            continue
        if s > last_e:
            if space_tokenize:
                # text_spans.extend(whitespace_tokenize(text[last_e: s]))
                # text_spans.extend(whitespace_tokenize_w_punctuation_ends(text[last_e: s]))
                text_spans.extend(word_tokenize(text[last_e: s]))
            else:
                tmp = text[last_e: s].strip()
                if tmp:
                    text_spans.append(tmp)
        indicate_mask = indicate_mask + [0] * (len(text_spans) - len(indicate_mask))

        text_spans.append(text[s: e].strip())
        indicate_mask = indicate_mask + [1] * (len(text_spans) - len(indicate_mask))
        last_e = e

    rest = text[last_e:].strip()
    if rest:
        if space_tokenize:
            # text_spans.extend(whitespace_tokenize(rest))
            # text_spans.extend(whitespace_tokenize_w_punctuation_ends(rest))
            text_spans.extend(word_tokenize(rest))
        else:
            text_spans.append(rest)
        indicate_mask = indicate_mask + [0] * (len(text_spans) - len(indicate_mask))

    # recovered_text = " ".join(text_spans)
    # if recovered_text != text:
    #     logger.warning(f"In consistent text during chunk:\n{recovered_text}\n{text}")
    #     print(f"In consistent text during chunk:\n{recovered_text}\n{text}")
    #     print(span_ls)
    #     print("======================")

    return text_spans, indicate_mask


def span_chunk_subword(text: str, span_ls: List[str]) -> Tuple[List[str], List[int]]:
    """
    Using the subword tokenization algorithm, e.g., BPR or wordpiece, to tokenize the sentence first,
    and find the span through recovery, which may have high time complexity.
    """
    pass


def span_chunk_simple(text: str, span_ls: List[str], tokenizer: PreTrainedTokenizer):
    """
    This version only process the entities spans and using pre-trained tokenizer to tokenize the text first
    to annotate the position of each span.
    """
    pos_ls = []
    for span in span_ls:
        span_pos_ls = find_span(text, span)
        pos_ls.extend(span_pos_ls)
    pos_ls = sorted(pos_ls, key=lambda x: x[0])

    for pos_id, pos in enumerate(pos_ls):
        if pos_id == 0:
            continue
        # assert pos[0] >= pos_ls[pos_id - 1][1], (span_ls, text[pos[0]: pos[1]], text[pos_ls[pos_id - 1][0]: pos_ls[pos_id - 1][1]])
        # There maybe bad case where a entity in a substring of another entity.
        # A bad case:
        # AssertionError: (['Netherlands', 'history of eindhoven', 'Koninkrijk der Nederlanden', 'Constituent country of the Kingdom of the Netherlands', 'Robert van der Horst', 'Eindhoven'],
        # 'Netherlands', 'Constituent country of the Kingdom of the Netherlands')
        if pos[0] < pos_ls[pos_id - 1][1]:
            return None, None

    tokens = []
    token_spans = []
    last_e = 0
    for s, e in pos_ls:
        if last_e > s:
            print(f"Overlapped span: {text[last_e: s]}\t{text[s: e]}\t{text}")
            continue

        sub_tokens = tokenizer.tokenize(text[last_e: s])
        find = False
        for a in range(len(sub_tokens)):
            if tokenizer.convert_tokens_to_string(sub_tokens[a:]).strip() == text[s: e]:
                find = True
                if a > 0:
                    tokens.extend(sub_tokens[:a])
                tk_s = len(tokens)
                tokens.extend(sub_tokens[a:])
                tk_e = len(tokens)
                token_spans.append((tk_s, tk_e))
                break

        if not find:
            while s - 1 >= last_e and text[s - 1] == ' ':
                s = s - 1  # To tokenize the space with the entity together.
            if s > last_e:
                tokens.extend(tokenizer.tokenize(text[last_e: s]))

            tk_s = len(tokens)
            tokens.extend(tokenizer.tokenize(text[s: e]))
            tk_e = len(tokens)
            token_spans.append((tk_s, tk_e))

        last_e = e

    if last_e < len(text):
        tokens.extend(tokenizer.tokenize(text[last_e:]))

    normalized_text = tokenizer.convert_tokens_to_string(tokens)

    # consistency check
    for s, e in token_spans:
        ent = tokenizer.convert_tokens_to_string(tokens[s: e]).strip()
        if ent not in span_ls:
            # print(f"Warning: {ent}\t{span_ls}")
            print(f"Warning: missed entity span after tokenization")
            return None, None

    _re_tokens = tokenizer.tokenize(normalized_text)
    if tokens != _re_tokens:
        # print(f"Warning: \n{tokens}\n{_re_tokens}\n{text}\n{normalized_text}")
        # print()
        # print(f"Warning: inconsistent tokens")
        return None, None
    if normalized_text != text:
        print(f"Warning, inconsistent text: {normalized_text}\t{text}")
        # return None, None

    return normalized_text, token_spans


def get_unused_tokens(_tokenizer: PreTrainedTokenizer, token_num: int = 4):
    if isinstance(_tokenizer, RobertaTokenizer) or isinstance(_tokenizer, RobertaTokenizerFast):
        _unused_token = "<unused{}>"
        _unused_tokens = []
        for i in range(token_num):
            _unused_tokens.append(_unused_token.format(str(i)))
        _tokenizer.add_tokens(_unused_tokens)
        return _unused_tokens
    elif isinstance(_tokenizer, AlbertTokenizer) or isinstance(_tokenizer, AlbertTokenizerFast):
        _unused_token = "[unused{}]"
        _unused_tokens = []
        for i in range(token_num):
            _unused_tokens.append(_unused_token.format(str(i)))
        _tokenizer.add_tokens(_unused_tokens)
        return _unused_tokens
    elif any([isinstance(_tokenizer, x) for x in [DebertaTokenizer, DebertaTokenizerFast, DebertaV2Tokenizer]]):
        _unused_token = "[unused{}]"
        _unused_tokens = []
        for i in range(token_num):
            _unused_tokens.append(_unused_token.format(str(i)))
        _tokenizer.add_tokens(_unused_tokens)
        return _unused_tokens


def dfs(src: List[int], vis: Set, state: List[int], ans: List[List[int]]):
    if len(state) == len(src):
        if not all(a == b for a, b in zip(src, state)):
            ans.append(state)

    for x in src:
        if x not in vis:
            new_vis = copy.deepcopy(vis)
            new_vis.add(x)
            new_state = copy.deepcopy(state)
            new_state.append(x)
            dfs(src, new_vis, new_state, ans)


def get_all_permutation(array: List[int]):
    res = []
    dfs(array, set(), list(), res)
    for state in res:
        assert not all(a == b for a, b in zip(state, array))
    return res


def recursive_find_path(node: Union[List, Dict, str], outputs: List[List[str]], res: List[str]):
    if isinstance(node, str):
        outputs.append(res + [node])
        return

    if isinstance(node, list):
        for x in node:
            recursive_find_path(x, outputs, res)
    elif isinstance(node, dict):
        for key, value in node.items():
            recursive_find_path(value, outputs, res + [key])
    else:
        raise ValueError('Unknown type: {}'.format(type(node)))


def recursive_bfs(deduction: Union[List, Dict]):
    res = ''

    queue = [deduction]
    while queue:
        node = queue.pop(0)
        if isinstance(node, str):
            res = res + ' ' + node
        elif isinstance(node, list):
            queue.extend(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                queue.append(value)
                res = res + ' ' + key
        else:
            raise ValueError('Unknown type: {}'.format(type(node)))

    return res.strip()


def dfs_enumerate_all_assign(keys: List[str], values: List[str], relation: str, res: List[str], assign: str,
                             key_vis: Set):
    if len(key_vis) == 0:
        res.append(assign)

    for key_id in key_vis:
        new_key_vis = copy.deepcopy(key_vis)
        new_key_vis.remove(key_id)
        for value in values:
            if value in keys[key_id]:
                continue
            new_assign = assign + ' ' + keys[key_id] + ' ' + relation + ' ' + value + '.'
            dfs_enumerate_all_assign(keys, values, relation, res, new_assign, new_key_vis)


def dfs_load_assignment(assignment_list, res: List[Tuple[str, str]], cur_assign: str):
    for assignment in assignment_list:
        if assignment['flag'] is False:
            continue
        if assignment['flag'] is None:
            res.append((cur_assign + ' ' + assignment['deduction'], assignment['id']))
        elif assignment['flag'] is True:
            dfs_load_assignment(assignment['assignment'], res, cur_assign + ' ' + assignment['deduction'])
        else:
            raise ValueError('Unknown flag: {}'.format(assignment['flag']))


def word_seq_to_word_char_starts(words: List[str]):
    """
    Args:
        words: The input word sequence (not subwords).
    """
    word2char_starts = []
    text = ""
    for word in words:
        if len(text) > 0:
            text = " " + text
        word2char_starts.append(len(text))
        text += word
    return word2char_starts, text


def char_to_subword_ids(text, tokenizer: PreTrainedTokenizer):
    subwords = tokenizer.tokenize(text)

    char2subword_ids = []
    char_lens = 0
    subword_idx = 0
    subwords_max_num = len(subwords)
    while subword_idx < subwords_max_num:
        subword_list = []
        prev_subword_idx = subword_idx
        subword_len = 0
        subword = ""
        while subword_idx < subwords_max_num:
            subword_list.append(subwords[subword_idx])
            subword_idx += 1
            subword = tokenizer.convert_tokens_to_string(subword_list)
            subword_len = len(subword)
            if subword == tokenizer.sep_token:
                char_lens += 1
            if text[char_lens: char_lens + subword_len] == subword:
                break
        assert text[char_lens: char_lens + subword_len] == subword
        if subword == "</s>":
            char2subword_ids.extend([prev_subword_idx] * (subword_len + 1))
        else:
            char2subword_ids.extend([prev_subword_idx] * subword_len)

        char_lens += len(subword)

    if len(text) != len(char2subword_ids):
        flag = False
    else:
        flag = True

    return char2subword_ids, subwords, flag
