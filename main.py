from rnn.correctText import correct_text
from utils import generateText, loadTXT

text = loadTXT('corpus/corpus_4.txt')

result = correct_text(text)

with open("result.txt", "w") as output_file:
    for i in range(0, len(result), 100):
        output_file.write(result[i:i + 100] + "\n")
output_file.close()
