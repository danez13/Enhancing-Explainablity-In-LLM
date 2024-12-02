"""Utilities extracting the annotated salient words from the datasets"""
from string import punctuation


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_gold_saliency_esnli(instance, tokens, special_tokens, tokenizer=None):
    gold_tokens = instance[3].lower().split(' ') + instance[4].lower().split(
        ' ')
    gold_tokens = [t for t in gold_tokens if len(t) > 0]
    in_gold_token = 0
    saliency_gold = []

    for token in tokens:
        token = token.replace('#', '')
        if token in special_tokens:
            saliency_gold.append(0)
            continue
        if token == gold_tokens[0]:
            saliency_gold.append(in_gold_token)
            gold_tokens.pop(0)
            continue

        if all(_t in punctuation for _t in gold_tokens):
            gold_tokens.pop(0)

        if gold_tokens[0].startswith('*') and len(gold_tokens[0]) == 1:
            in_gold_token = 0
            gold_tokens.pop(0)

        if gold_tokens[0].startswith('*') and len(gold_tokens[0]) > 1:
            in_gold_token = 1
            gold_tokens[0] = gold_tokens[0][1:]

        if gold_tokens[0].startswith(token):
            saliency_gold.append(in_gold_token)
            gold_tokens[0] = gold_tokens[0][len(token):]
            if gold_tokens[0] == '*':
                gold_tokens.pop(0)
                in_gold_token = 0

        else:
            print('OOOPs', token)
            saliency_gold.append(0)

    return saliency_gold
