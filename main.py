from utils import *
import os

CORPUS = os.environ.get('CORPUS')


def generateText(starting_sent, k=4, maxLen=1000):
    sentence = starting_sent
    ctx = starting_sent[-k:]
    model = MarkovChain(loadTXT('{}/corpus.txt'.format(CORPUS)))
    for ix in range(maxLen):
        next_prediction = sampleNext(ctx, model, k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence


print("Function Created Successfully!")

text = generateText("dear", k=4, maxLen=2000)
print(text)
