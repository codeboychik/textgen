import transformers
import spellchecker
import torch as torch
import re
# Load the pre-trained language model and the spell checker
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
lm_model = transformers.AutoModelForMaskedLM.from_pretrained('bert-base-cased')
spell = spellchecker.SpellChecker()


def correct_text(text):
    # Tokenize the text
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True)

    # Generate candidate corrections for each mistake
    with torch.no_grad():
        outputs = lm_model(input_ids)
        predictions = outputs[0]
        predicted_tokens = torch.argmax(predictions, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(predicted_tokens)
    mistakes = [token for i, token in enumerate(tokens) if token.startswith('##')]
    corrections = [spell.correction(token[2:]) for token in mistakes]

    # Apply the corrections to the original text
    corrected_tokens = []
    prev_token = None
    for token in tokens:
        corrected_token = ''
        if token == prev_token:
            continue
        if token in mistakes:
            corrected_token = spell.correction(token[2:])
            if corrected_token == prev_token:
                continue
            corrected_tokens.append(corrected_token)
        else:
            corrected_tokens.append(token)
        prev_token = corrected_token
    corrected_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(corrected_tokens))

    # Remove consecutive dots and colons
    corrected_text = re.sub(r'([.:])\1+', r'\1', corrected_text)
    corrected_text = re.sub(r'(\.)\1+', '.', corrected_text)
    corrected_text = re.sub(r'\b[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]+\b| \[UNK\] ', '', corrected_text)

    # Capitalize the first letter of each sentence
    corrected_text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s*([a-z])', lambda match: match.group(0).upper(), corrected_text)

    # Capitalize the pronoun "I"
    corrected_text = re.sub(r'\bi\b', 'I', corrected_text)
    return corrected_text
