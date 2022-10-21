from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
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
    print(len(train_dataset), train_dataset[0])

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(args.device)

    max_seq_length = 512
    num_proc = 4

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
    )
    print(tokenized_dataset.shape)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together,
    # so group_texts throws away a remainder for each of those groups of 1,000 texts.
    # You can adjust that batch_size here but a higher value might be slower to preprocess.

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
    )
    print(tokenized_dataset.shape)
    print(tokenized_dataset.data)
    print(tokenized_dataset[0])

    input_ids = tokenizer.encode("This is", return_tensors="pt")
    generations = model.generate(input_ids=input_ids.to(args.device))
    for gen in generations:
        new_text = tokenizer.decode(gen)
        print(new_text)

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
