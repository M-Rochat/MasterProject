from torch.utils.data import TensorDataset
import evaluate
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, pipeline
from argparse import ArgumentParser
import torch
from torch import cuda
import numpy as np
from datasets import load_dataset, DownloadMode


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=42)
    parser.add_argument("--model_name", default='facebook/bart-base')
    parser.add_argument("--dataset_dir", default="modified_dataset/")
    parser.add_argument("--reset_cache", action='store_true')
    parser.add_argument("--device", default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument("--output_dir", default="./seq_to_seq", help="The output directory")
    parser.add_argument("--overwrite_output_dir", default=True, help=" overwrite the content of the output directory")
    parser.add_argument("--num_train_epochs", default=3)  # number of training epochs
    parser.add_argument("--per_device_train_batch_size", default=32)  # batch size for training
    parser.add_argument("--per_device_eval_batch_size", default=64)  # batch size for evaluation
    parser.add_argument("--eval_steps", default=400)  # Number of update steps between two evaluations.
    parser.add_argument("--save_steps", default=800)  # after # steps model is saved
    parser.add_argument("--warmup_steps", default=500)  # number of warmup steps for learning rate scheduler
    parser.add_argument("--prediction_loss_only", default=True)
    args = parser.parse_args(args=[])


    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    
    data_files = {"train": args.dataset_dir + "train.tsv", "test": args.dataset_dir + "test.tsv",
                  "dev": args.dataset_dir + "dev.tsv"}
    dataset = load_dataset('csv', data_files=data_files, delimiter='\t',
                           download_mode=DownloadMode.FORCE_REDOWNLOAD if args.reset_cache else DownloadMode.REUSE_DATASET_IF_EXISTS)
    print("Dataset loaded", len(dataset['train']), dataset['train'][0])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = "[PAD]"
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(args.device)
    print("Model + tokenizer downloaded")

    max_seq_length = 64

    def preprocess_function(examples):
        model_inputs = tokenizer(examples['head'], text_target=examples['tail'], max_length=max_seq_length,
                                 truncation=True)
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,  # num_proc=num_proc,
        remove_columns=['head', 'tail'],
        load_from_cache_file=True
    )

    print('Tokenization done')
    print(tokenized_dataset['train'].shape)
    print(tokenized_dataset['train'][0])

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )

    batch = data_collator([tokenized_dataset["train"][i] for i in range(1, 3)])
    print(batch.keys())

    # Metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     # if data_args.ignore_pad_token_for_loss:
    #     #     # Replace -100 in the labels as we can't decode them.
    #     #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #     # Some simple post-processing
    #     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    #
    #     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #     result = {k: round(v * 100, 4) for k, v in result.items()}
    #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     return result

    trainer = Seq2SeqTrainer(
        model=model,
        # args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Test model
    input_ids = tokenizer.encode("I give you an apple. I am", return_tensors="pt")
    generations = model.generate(input_ids=input_ids.to(args.device))
    for gen in generations:
        new_text = tokenizer.decode(gen)
        print(new_text)


if __name__ == '__main__':
    main()
