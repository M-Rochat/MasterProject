from transformers import RobertaTokenizer, RobertaModel


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)
# from transformers import TrainingArguments, Trainer
#
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
# trainer = Trainer(
#
#     model=model,
#
#     args=training_args,
#
#     train_dataset=small_train_dataset,
#
#     eval_dataset=small_eval_dataset,
#
#     compute_metrics=compute_metrics,
#
# )
#
# def compute_metrics(eval_pred):
#
#     logits, labels = eval_pred
#
#     predictions = np.argmax(logits, axis=-1)
#
#     return metric.compute(predictions=predictions, references=labels)