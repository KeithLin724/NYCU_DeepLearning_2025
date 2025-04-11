import json
import string
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

with open("/home/Result/submission.json", "r", encoding="utf-8") as f:
    inference_results = json.load(f)

references = []
hypotheses = []

smoothing_function = SmoothingFunction().method1

unique_indices = set()

for item in tqdm(inference_results, total=50):
    idx = item["idx"]

    if idx in unique_indices:
        print("idx {} already processed.".format(idx))
        break

    unique_indices.add(idx)
    
    generated_caption = normalize_text(item["output"])
    standard_caption = normalize_text(TA_dataset[idx]["caption"])
    
    references.append([standard_caption.split()])
    hypotheses.append(generated_caption.split())

corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing_function)
print("Corpus BLEU score: {:.4f}".format(corpus_bleu_score))