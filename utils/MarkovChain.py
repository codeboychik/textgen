from utils import generateTable, convertFreqIntoProb


def MarkovChain(text, k=4):
    T = generateTable(text, k)
    T = convertFreqIntoProb(T)
    return T


print('Model Created Successfully!')