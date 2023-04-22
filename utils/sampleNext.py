import numpy as np


def sampleNext(ctx, model, k):
    ctx = ctx[-k:]
    if model.get(ctx) is None:
        return " "
    possible_Chars = list(model[ctx].keys())
    possible_values = list(model[ctx].values())
    return np.random.choice(possible_Chars, p=possible_values)




