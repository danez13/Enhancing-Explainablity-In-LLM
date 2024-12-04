"""Utilities extracting the annotated salient words from the datasets"""
from string import punctuation  # Importing punctuation characters to handle in the text processing

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get  # Allows attribute-style access to dictionary keys (e.g., obj.key instead of obj['key'])
    __setattr__ = dict.__setitem__  # Allows setting values using attribute-style syntax (e.g., obj.key = value)
    __delattr__ = dict.__delitem__  # Allows deletion of dictionary items using attribute-style syntax (e.g., del obj.key)


def get_gold_saliency_esnli(instance, tokens, special_tokens, tokenizer=None):
    """
    Function to extract saliency of gold tokens for a given instance of data in the ESNLI dataset.
    It compares the tokens in the input with the "gold" tokens and assigns a saliency score.
    
    Parameters:
    - instance: A list or tuple where element 3 and 4 contain the gold tokens (target words).
    - tokens: A list of tokens to be processed.
    - special_tokens: A list of tokens that are special and should be excluded from saliency calculation.
    - tokenizer: An optional tokenizer (not used in this version).
    
    Returns:
    - saliency_gold: A list indicating the saliency of each token in the input `tokens` list.
    """
    # Split the gold tokens (target words) from the instance, convert them to lowercase, and clean up any empty tokens
    gold_tokens = instance[3].lower().split(' ') + instance[4].lower().split(' ')
    gold_tokens = [t for t in gold_tokens if len(t) > 0]  # Remove any empty tokens
    in_gold_token = 0  # Variable to track whether the current token is part of the gold tokens
    saliency_gold = []  # List to store saliency scores for each token

    # Loop through each token in the input tokens list
    for token in tokens:
        token = token.replace('#', '')  # Remove any '#' from the token (to clean up)
        
        if token in special_tokens:  # If token is a special token, assign a saliency of 0
            saliency_gold.append(0)
            continue  # Skip to the next token
        
        if token == gold_tokens[0]:  # If token matches the current gold token, assign saliency based on `in_gold_token`
            saliency_gold.append(in_gold_token)
            gold_tokens.pop(0)  # Remove the matched gold token from the list
            continue
        
        # If all the remaining gold tokens are punctuation, pop the first gold token
        if all(_t in punctuation for _t in gold_tokens):
            gold_tokens.pop(0)
        
        # Handle special cases for gold tokens that start with '*'
        if gold_tokens[0].startswith('*') and len(gold_tokens[0]) == 1:
            in_gold_token = 0  # Reset saliency if gold token is just '*'
            gold_tokens.pop(0)
        
        if gold_tokens[0].startswith('*') and len(gold_tokens[0]) > 1:
            in_gold_token = 1  # Mark the gold token as part of the saliency
            gold_tokens[0] = gold_tokens[0][1:]  # Remove '*' from the start of the token

        # If the current token matches the start of the gold token, update saliency
        if gold_tokens[0].startswith(token):
            saliency_gold.append(in_gold_token)
            gold_tokens[0] = gold_tokens[0][len(token):]  # Remove the matched part of the gold token
            
            # If the gold token becomes '*', pop it
            if gold_tokens[0] == '*':
                gold_tokens.pop(0)
                in_gold_token = 0  # Reset saliency for next token

        else:
            # If token does not match the expected gold token, append 0 (no saliency)
            print('OOOPs', token)
            saliency_gold.append(0)

    return saliency_gold  # Return the list of saliency scores for the input tokens
