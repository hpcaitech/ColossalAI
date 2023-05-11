import jieba
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
from bert_score import score
from sklearn.metrics import precision_score, recall_score, f1_score


def bleu_score(preds: list, targets: list) -> dict:
    bleu_score = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    cumulate_bleu1 = cumulate_bleu2 = cumulate_bleu3 = cumulate_bleu4 = 0

    # calculate scores
    for pred, target in zip(preds, targets):
        pred_list = []
        target_list = []
        pred_list = (' '.join(jieba.cut(pred))).split()
        target_list.append((' '.join(jieba.cut(target))).split())

        bleu1 = sentence_bleu(target_list, pred_list, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu(target_list, pred_list, weights=(0, 1, 0, 0))
        bleu3 = sentence_bleu(target_list, pred_list, weights=(0, 0, 1, 0))
        bleu4 = sentence_bleu(target_list, pred_list, weights=(0, 0, 0, 1))

        cumulate_bleu1 += bleu1
        cumulate_bleu2 += bleu2
        cumulate_bleu3 += bleu3
        cumulate_bleu4 += bleu4

    bleu_score["bleu1"] = cumulate_bleu1 / len(preds)
    bleu_score["bleu2"] = cumulate_bleu2 / len(preds)
    bleu_score["bleu3"] = cumulate_bleu3 / len(preds)
    bleu_score["bleu4"] = cumulate_bleu4 / len(preds)

    return bleu_score


def rouge_score(preds: list, targets: list) -> dict:
    rouge_scores = {"rouge1": {}, "rouge2": {}, "rougeL": {}}

    # calculate scores
    preds_all = []
    targets_all = []

    for pred, target in zip(preds, targets):
        pred_list = ' '.join(jieba.cut(pred))
        target_list = ' '.join(jieba.cut(target))
        preds_all.append(pred_list)
        targets_all.append(target_list)

    rouge = Rouge()
    rouge_avg = rouge.get_scores(preds_all, targets_all, avg=True)
    rouge_scores["rouge1"] = rouge_avg["rouge-1"]
    rouge_scores["rouge2"] = rouge_avg["rouge-2"]
    rouge_scores["rougeL"] = rouge_avg["rouge-l"]

    return rouge_scores


def distinct_score(preds: list) -> dict:
    distinct_score = {"distinct": 0}
    cumulate_distinct = 0

    # calculate scores
    for pred in preds:
        pred_list = ' '.join(jieba.cut(pred))
        chars = list(pred_list)
        count_chars = len(chars)
        unique_chars = set(chars)
        count_unique_chars = len(unique_chars)
        cumulate_distinct += count_unique_chars / count_chars

    distinct_score["distinct"] = cumulate_distinct / len(preds)

    return distinct_score


def bert_score(preds: list, targets: list) -> dict:
    bert_score = {"precision": 0, "recall": 0, "f1_score": 0}
    f1_scores = 0
    recall_scores = 0
    precision_scores = 0

    # calculate scores
    for pred, target in zip(preds, targets):
        pred_list = []
        target_list = []
        pred_list.append(' '.join(jieba.cut(pred)))
        target_list.append(' '.join(jieba.cut(target)))
        P, R, F1 = score(pred_list, target_list, lang="zh", verbose=True)
        precision_scores += P
        recall_scores += R
        f1_scores += F1

    bert_score["precision"] = precision_scores / len(preds)
    bert_score["recall"] = recall_scores / len(preds)
    bert_score["f1_score"] = f1_scores / len(preds)

    return bert_score


def precision(preds: list, targets: list) -> dict:
    precision = {"precision": 0}
    precision_scores = 0

    # calculate scores
    for pred, target in zip(preds, targets):
        pred_list = [char for char in pred]
        target_list = [char for char in target]

        target_labels = [1] * len(target_list)
        pred_labels = [int(pred_list[i] == target_list[i]) for i in range(0, len(target_list))]
        precision_scores += precision_score(target_labels, pred_labels)

    precision["precision"] = precision_scores / len(preds)

    return precision


def recall(preds: list, targets: list) -> dict:
    recall = {"recall": 0}
    recall_scores = 0

    # calculate scores
    for pred, target in zip(preds, targets):
        pred_list = [char for char in pred]
        target_list = [char for char in target]

        target_labels = [1] * len(target_list)
        pred_labels = [int(pred_list[i] == target_list[i]) for i in range(0, len(target_list))]
        recall_scores += recall_score(target_labels, pred_labels)

    recall["recall"] = recall_scores / len(preds)

    return recall


def f1_score(preds: list, targets: list) -> dict:
    f1 = {"f1_score": 0}
    f1_scores = 0

    # calculate scores
    for pred, target in zip(preds, targets):
        pred_list = [char for char in pred]
        target_list = [char for char in target]

        target_labels = [1] * len(target_list)
        pred_labels = [int(pred_list[i] == target_list[i]) for i in range(0, len(target_list))]
        f1_scores += f1_score(target_labels, pred_labels)

    f1["f1_score"] = f1_scores / len(preds)

    return f1
