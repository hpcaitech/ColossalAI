import statistics

import jieba
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge as Rouge_cn
from sklearn.metrics import f1_score, precision_score, recall_score


def bleu_score(preds: list, targets: list) -> dict:
    """Calculate BLEU Score Metric

    The calculation includes BLEU-1 for unigram, BLEU-2 for bigram,
    BLEU-3 for trigram and BLEU-4 for 4-gram. Unigram evaluates the
    accuracy in word level, other n-gram evaluate the fluency in
    sentence level.
    """
    bleu_scores = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    cumulative_bleu = [0] * 4
    weights = [(1. / 1., 0., 0., 0.), (1. / 2., 1. / 2., 0., 0.), (1. / 3., 1. / 3., 1. / 3., 0.),
               (1. / 4., 1. / 4., 1. / 4., 1. / 4.)]

    for pred, target in zip(preds, targets):
        pred_list = (' '.join(jieba.cut(pred))).split()
        target_list = [(' '.join(jieba.cut(target))).split()]

        bleu = sentence_bleu(target_list, pred_list, weights=weights)
        cumulative_bleu = [a + b for a, b in zip(cumulative_bleu, bleu)]

    for i in range(len(cumulative_bleu)):
        bleu_scores[f"bleu{i+1}"] = cumulative_bleu[i] / len(preds)

    return bleu_scores


def rouge_cn_score(preds: list, targets: list) -> dict:
    """Calculate Chinese ROUGE Score Metric

    The calculation includes ROUGE-1 for unigram, ROUGE-2 for bigram
    and ROUGE-L. ROUGE-N evaluates the number of matching n-grams between
    the preds and targets. ROUGE-L measures the number of matching
    longest common subsequence (LCS) between preds and targets.
    """
    rouge_scores = {"rouge1": {}, "rouge2": {}, "rougeL": {}}
    all_preds = []
    all_targets = []

    for pred, target in zip(preds, targets):
        pred_list = ' '.join(jieba.cut(pred))
        target_list = ' '.join(jieba.cut(target))
        all_preds.append(pred_list)
        all_targets.append(target_list)

    rouge_cn = Rouge_cn()
    rouge_avg = rouge_cn.get_scores(all_preds, all_targets, avg=True)

    rouge_scores["rouge1"] = rouge_avg["rouge-1"]["f"]
    rouge_scores["rouge2"] = rouge_avg["rouge-2"]["f"]
    rouge_scores["rougeL"] = rouge_avg["rouge-l"]["f"]

    return rouge_scores


def distinct_score(preds: list) -> dict:
    """Calculate Distinct Score Metric

    This metric refers to https://arxiv.org/abs/1510.03055.
    It evaluates the diversity of generation text by counting
    the unique n-grams.
    """
    distinct_score = {"distinct": 0}
    cumulative_distinct = []

    for pred in preds:
        pred_seg_list = list(' '.join(jieba.cut(pred)))
        count_segs = len(pred_seg_list)
        unique_segs = set(pred_seg_list)
        count_unique_chars = len(unique_segs)

        cumulative_distinct.append(count_unique_chars / count_segs)

    distinct_score["distinct"] = statistics.mean(cumulative_distinct)

    return distinct_score


def bert_score(preds: list, targets: list) -> dict:
    """Calculate BERTScore Metric

    The BERTScore evaluates the semantic similarity between
    tokens of preds and targets with BERT.
    """
    bert_score = {"bert_score": 0}
    pred_list = []
    target_list = []

    for pred, target in zip(preds, targets):
        pred_list.append(' '.join(jieba.cut(pred)))
        target_list.append(' '.join(jieba.cut(target)))

    _, _, F = score(pred_list, target_list, lang="zh", verbose=True)

    bert_score["bert_score"] = F.mean().item()

    return bert_score


def calculate_precision_recall_f1(preds: list, targets: list) -> dict:
    """Precision, Recall and F1-Score Calculation

    The calculation of precision, recall and f1-score is realized by counting
    the number f overlaps between the preds and target. The comparison length
    limited by the shorter one of preds and targets. This design is mainly
    considered for classifiction and extraction categories.
    """
    precision_recall_f1 = {"precision": 0, "recall": 0, "f1_score": 0}
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, target in zip(preds, targets):
        pred_list = [char for char in pred]
        target_list = [char for char in target]

        target_labels = [1] * min(len(target_list), len(pred_list))
        pred_labels = [int(pred_list[i] == target_list[i]) for i in range(0, min(len(target_list), len(pred_list)))]

        precision_scores.append(precision_score(target_labels, pred_labels, zero_division=0))
        recall_scores.append(recall_score(target_labels, pred_labels, zero_division=0))
        f1_scores.append(f1_score(target_labels, pred_labels, zero_division=0))

    precision_recall_f1["precision"] = statistics.mean(precision_scores)
    precision_recall_f1["recall"] = statistics.mean(recall_scores)
    precision_recall_f1["f1_score"] = statistics.mean(f1_scores)

    return precision_recall_f1


def precision(preds: list, targets: list) -> dict:
    """Calculate Precision Metric
    (design for classifiction and extraction categories)

    Calculating precision by counting the number of overlaps between the preds and target.
    """
    precision = {"precision": 0}
    precision["precision"] = calculate_precision_recall_f1(preds, targets)["precision"]
    return precision


def recall(preds: list, targets: list) -> dict:
    """Calculate Recall Metric
    (design for classifiction and extraction categories)

    Calculating recall by counting the number of overlaps between the preds and target.
    """
    recall = {"recall": 0}
    recall["recall"] = calculate_precision_recall_f1(preds, targets)["recall"]
    return recall


def F1_score(preds: list, targets: list) -> dict:
    """Calculate F1-score Metric
    (design for classifiction and extraction categories)

    Calculating f1-score by counting the number of overlaps between the preds and target.
    """
    f1 = {"f1_score": 0}
    f1["f1_score"] = calculate_precision_recall_f1(preds, targets)["f1_score"]
    return f1
