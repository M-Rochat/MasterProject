from nltk.translate.bleu_score import sentence_bleu
"Which color is the sky?"
references = ["Blue","Blue with white clouds", "Dark and menacing"]
hypothesis = "Blue dark"
score=sentence_bleu(references=references,hypothesis= hypothesis)
print(score)