def loadTXT(filename):
    with open(filename, encoding='utf8') as f:
        return f.read().lower()


print('Loaded the dataset.')
