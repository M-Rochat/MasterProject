import pandas as pd
import inflect
import spacy
from argparse import ArgumentParser
from matplotlib import pyplot, colors as mcolors

inflection_engine = inflect.engine()
nlp = spacy.load("en_core_web_sm")


def triplet_to_text(fact):
    """
    Take a dict ['head','relation','tail'] and update 'head' and 'tail' in a text form
    @param fact:
    """

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


def plot_bar(data, relations):
    sum = 0
    count = []
    labels = []
    colors = []
    kind_to_colors = {"Physical-Entities": 'b', "Event-Centered": 'g', "Social-Interacton": 'r'}
    for kind, relations in relations.items():
        colors += kind_to_colors[kind] * len(relations)
        for r in relations:
            labels.append(r)
            size = data.loc[data["relation"] == r].shape[0]
            count.append(size)
            sum += size
    assert sum == data.shape[0], "Dataset contains unknown relations"
    pyplot.bar(x=range(len(labels)), height=count, color=colors, label=labels)
    pyplot.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default='dataset/')
    parser.add_argument("--output_dir", default="modified_dataset/")
    parser.add_argument("--eval", default=False)
    args = parser.parse_args()

    for split in ("train", "test", "dev"):
        dataset_dir = args.dataset_dir + split + ".tsv"
        output_dir = args.output_dir + split + ".tsv"
        data = pd.read_csv(dataset_dir, sep='\t', names=["head", "relation", "tail"])
        relations = {"Physical-Entities": ["ObjectUse", "CapableOf", "MadeUpOf", "HasProperty", "Desires", "NotDesires",
                                           "AtLocation"],
                     "Event-Centered": ["Causes", "HinderedBy", "xReason", "isAfter", "isBefore", "HasSubEvent",
                                        "isFilledBy"],
                     "Social-Interacton": ["xIntent", "xReact", "oReact", "xAttr", "xEffect", "xNeed", "xWant",
                                           "oEffect",
                                           "oWant"]}

        # plot_bar(data, relations)
        print(data.shape)
        for word in ('none', 'None', 'NONE', 'NONEQ'):
            data = data[data["tail"] != word]
            print(data.shape)
        data = data[data["tail"].notna()]
        print(data.shape)

        data.apply(triplet_to_text, axis="columns")
        data.drop(columns="relation", inplace=True)
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle the dataframe

        data.to_csv(output_dir, index=None, sep='\t')


if __name__ == '__main__':
    main()
