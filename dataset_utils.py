import pandas as pd
import inflect
import spacy
from argparse import ArgumentParser


def triplet_to_text(fact):
    """
    Take a dict ['head','relation','tail'] and update 'head' and 'tail' in a text form
    @param fact:
    """

    inflection_engine = inflect.engine()
    nlp = spacy.load("en_core_web_sm")

    def article(word):
        return "an" if word[0] in ['a', 'e', 'i', 'o', 'u'] else "a"

    def vp_present_participle(phrase):
        doc = nlp(phrase)
        return ' '.join([
            inflection_engine.present_participle(
                token.text) if token.pos_ == "VERB" and token.tag_ != "VGG" else token.text
            for token in doc
        ])

    head = fact['head']
    relation = fact['relation']
    tail = str(fact['tail'])

    tail.strip()
    if tail.lower()[:3] == "to ":
        tail = tail[2:]
        tail.strip()

    if relation == "AtLocation":
        prompt = "You are likely to find {} {} in {} ".format(
            article(head), head, article(tail)
        )
    elif relation == "CapableOf":
        prompt = "{} can ".format(head)
    elif relation == "Causes":
        prompt = "Sometimes {} causes ".format(head)
    elif relation == "Desires":
        prompt = "{} {} desires".format(article(head), head)
    elif relation == "HasProperty":
        prompt = "{} is ".format(head)
    elif relation == "HasSubEvent":
        prompt = "While {}, you would ".format(vp_present_participle(head))
    elif relation == "HinderedBy":
        prompt = "{}. This would not happen if".format(head)
    elif relation == "MadeUpOf":
        prompt = "{} {} contains".format(article(head), head)
    elif relation == "NotDesires":
        prompt = "{} {} does not desire".format(article(head), head)
    elif relation == "ObjectUse":
        prompt = "{} {} can be used to".format(article(head), head)
    elif relation == "isAfter":
        prompt = "{}. Before that, ".format(head)
    elif relation == "isBefore":
        prompt = "{}. After that, ".format(head)
    elif relation == "isFilledBy":
        prompt = "{} is filled by".format(head)
    elif relation == "oEffect":
        prompt = "{}. The effect on others will be that others".format(head)
    elif relation == "oReact":
        prompt = "{}. As a result, others feel".format(head)
    elif relation == "oWant":
        prompt = "{}. After, others will want to".format(head)
    elif relation == "xAttr":
        prompt = "{}. PersonX is".format(head)
    elif relation == "xEffect":
        prompt = "{}. The effect on PersonX will be that PersonX".format(head)
    elif relation == "xIntent":
        prompt = "{}. PersonX did this to".format(head)
    elif relation == "xNeed":
        prompt = "{}. Before, PersonX needs to".format(head)
    elif relation == "xReact":
        prompt = "{}. PersonX will be".format(head)
    elif relation == "xReason":
        prompt = "{}. PersonX did this because".format(head)
    elif relation == "xWant":
        prompt = "{}. After, PersonX will want to".format(head)
    else:
        raise Exception(relation)
    fact['head'] = prompt.strip()
    fact['tail'] = tail


def count_examples(data, relations):
    sum = 0
    for r in relations:
        size = data.loc[data["relation"] == r].shape[0]
        print(r, size)
        sum += size
    print(sum, "/", data.shape[0])

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default='dataset/train.tsv')
    parser.add_argument("--output_dir", default="modified_dataset/train")
    parser.add_argument("--eval", default=False)
    args = parser.parse_args()


    data = pd.read_csv(args.dataset_dir, sep='\t', names=["head", "relation", "tail"])
    relations = ["ObjectUse", "CapableOf", "MadeUpOf", "HasProperty", "Desires", "NotDesires", "AtLocation"] + \
                ["Causes", "HinderedBy", "xReason", "isAfter", "isBefore", "HasSubEvent", "isFilledBy"] \
                + ["xIntent", "xReact", "oReact", "xAttr", "xEffect", "xNeed", "xWant", "oEffect", "oWant"]

    count_examples(data, relations)

    data.drop(data[data["tail"] == "none"].index, inplace=True)
    data.apply(triplet_to_text, axis="columns")
    data.drop(columns="relation", inplace=True)
    data = data.sample(frac=1).reset_index(drop=True) #shuffle the dataframe

    if args.eval:
        pass
    else:
        data["text"]= data["head"]+ " " + data["tail"]
        data.drop(columns=["tail","head"],inplace=True)
    data.to_json(args.output_dir+".json")
    data.to_csv(args.output_dir+".csv", index=None, sep='\t')


if __name__ == '__main__':
    main()
    #data = pd.read_csv("modified_dataset/train.csv", sep='\t')
    #data.to_json("modified_dataset/train.json")
