from torch.utils.data import TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, Seq2SeqTrainer
from argparse import ArgumentParser
import torch
from torch import cuda
import numpy as np
from datasets import load_dataset, DownloadMode


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=42)
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset_dir", default="./modified_dataset/train.csv")
    parser.add_argument("--reset_cache", action='store_true')
    parser.add_argument("--device", default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument("--output_dir", default="./gpt2-dir", help="The output directory")
    parser.add_argument("--overwrite_output_dir", default=True, help=" overwrite the content of the output directory")
    parser.add_argument("--num_train_epochs", default=3)  # number of training epochs
    parser.add_argument("--per_device_train_batch_size", default=32)  # batch size for training
    parser.add_argument("--per_device_eval_batch_size", default=64)  # batch size for evaluation
    parser.add_argument("--eval_steps", default=400)  # Number of update steps between two evaluations.
    parser.add_argument("--save_steps", default=800)  # after # steps model is saved
    parser.add_argument("--warmup_steps", default=500)  # number of warmup steps for learning rate scheduler
    parser.add_argument("--prediction_loss_only", default=True)
    args = parser.parse_args()

    # training_args = TrainingArguments(*args)

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed

    train_dataset = load_dataset('csv', data_files=args.dataset_dir, delimiter='\t',
                                 download_mode=DownloadMode.FORCE_REDOWNLOAD if args.reset_cache else DownloadMode.REUSE_DATASET_IF_EXISTS)[
        'train']

    print("Dataset loaded",len(train_dataset), train_dataset[0])

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token="[PAD]"
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(args.device)
    for example in train_dataset['tail']:
        if not isinstance(example,str):
            print(example)

    max_seq_length = 512
    num_proc = 4

    def tokenize_function(examples):
        model_inputs = tokenizer(examples['head'], max_length=max_seq_length, pad_to_max_length=True, truncation=True)
        labels = tokenizer(text_target=examples['tail'], max_length=max_seq_length, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        #     labels["input_ids"] = [
        #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,#num_proc=num_proc,
    )

    print(tokenized_dataset.shape)
    print(tokenized_dataset.data)
    print(tokenized_dataset[0])

    input_ids = tokenizer.encode("This is", return_tensors="pt")
    generations = model.generate(input_ids=input_ids.to(args.device))
    for gen in generations:
        new_text = tokenizer.decode(gen)
        print(new_text)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # if data_args.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=None,
    #     tokenizer=tokenizer,
    #     # Data collator will default to DataCollatorWithPadding, so we change it.
    #     data_collator=default_data_collator,
    #     compute_metrics=compute_metrics if training_args.do_eval else None,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics
    #     if training_args.do_eval and not is_torch_tpu_available()
    #     else None,
    # )


if __name__ == '__main__':
    main()
