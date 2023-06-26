import statistics
from typing import Dict, List

import jieba
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from rouge_chinese import Rouge as Rouge_cn
from rouge_score import rouge_scorer as Rouge_en
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import preprocessing_text, remove_redundant_space


def bleu_score(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
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
        if language == "cn":
            pred_list = ' '.join(jieba.cut(preprocessing_text(pred))).split()
            target_list = [(' '.join(jieba.cut(preprocessing_text(target)))).split()]
        elif language == "en":
            pred_list = preprocessing_text(pred).split()
            target_list = [preprocessing_text(target).split()]

        bleu = sentence_bleu(target_list, pred_list, weights=weights)
        cumulative_bleu = [a + b for a, b in zip(cumulative_bleu, bleu)]

    for i in range(len(cumulative_bleu)):
        bleu_scores[f"bleu{i+1}"] = cumulative_bleu[i] / len(preds)

    return bleu_scores


def chrf_score(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Calculate CHRF Score Metric in sentence level.
    """
    chrf_score = {"chrf": 0}
    cumulative_chrf = []

    for pred, target in zip(preds, targets):
        if language == "cn":
            pred_list = ' '.join(jieba.cut(preprocessing_text(pred))).split()
            target_list = ' '.join(jieba.cut(preprocessing_text(target))).split()
        elif language == "en":
            pred_list = preprocessing_text(pred).split()
            target_list = preprocessing_text(target).split()

        cumulative_chrf.append(sentence_chrf(target_list, pred_list))

    chrf_score["chrf"] = statistics.mean(cumulative_chrf)

    return chrf_score


def rouge_cn_score(preds: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate Chinese ROUGE Score Metric

    The calculation includes ROUGE-1 for unigram, ROUGE-2 for bigram
    and ROUGE-L. ROUGE-N evaluates the number of matching n-grams between
    the preds and targets. ROUGE-L measures the number of matching
    longest common subsequence (LCS) between preds and targets.
    """
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    all_preds = []
    all_targets = []

    for pred, target in zip(preds, targets):
        pred_list = remove_redundant_space(' '.join(jieba.cut(preprocessing_text(pred))))
        target_list = remove_redundant_space(' '.join(jieba.cut(preprocessing_text(target))))
        all_preds.append(pred_list)
        all_targets.append(target_list)

    rouge_cn = Rouge_cn()
    rouge_avg = rouge_cn.get_scores(all_preds, all_targets, avg=True)

    rouge_scores["rouge1"] = rouge_avg["rouge-1"]["f"]
    rouge_scores["rouge2"] = rouge_avg["rouge-2"]["f"]
    rouge_scores["rougeL"] = rouge_avg["rouge-l"]["f"]

    return rouge_scores


def rouge_en_score(preds: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate English ROUGE Score Metric

    The calculation includes ROUGE-1 for unigram, ROUGE-2 for bigram
    and ROUGE-L. ROUGE-N evaluates the number of matching n-grams between
    the preds and targets. ROUGE-L measures the number of matching
    longest common subsequence (LCS) between preds and targets.
    """
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    all_preds = []
    all_targets = []

    rouge_en = Rouge_en.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    for pred, target in zip(preds, targets):
        score = rouge_en.score(preprocessing_text(pred), preprocessing_text(target))
        rouge_scores["rouge1"] += score['rouge1'].fmeasure
        rouge_scores["rouge2"] += score['rouge2'].fmeasure
        rouge_scores["rougeL"] += score['rougeL'].fmeasure

    rouge_scores["rouge1"] = rouge_scores["rouge1"] / len(preds)
    rouge_scores["rouge2"] = rouge_scores["rouge2"] / len(preds)
    rouge_scores["rougeL"] = rouge_scores["rougeL"] / len(preds)

    return rouge_scores


def rouge_score(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Calculate ROUGE Score Metric"""
    if language == "cn":
        return rouge_cn_score(preds, targets)
    elif language == "en":
        return rouge_en_score(preds, targets)


def distinct_score(preds: List[str], language: str) -> Dict[str, float]:
    """Calculate Distinct Score Metric

    This metric refers to https://arxiv.org/abs/1510.03055.
    It evaluates the diversity of generation text by counting
    the unique n-grams.
    """
    distinct_score = {"distinct": 0}
    cumulative_distinct = []

    for pred in preds:
        if language == "cn":
            pred_seg_list = ' '.join(jieba.cut(pred)).split()
            count_segs = len(pred_seg_list)
            unique_segs = set(pred_seg_list)
            count_unique_chars = len(unique_segs)
            # prevent denominator from being 0
            cumulative_distinct.append(count_unique_chars / (count_segs + 1e-6))
        elif language == "en":
            # calculate distinct 1-gram, 2-gram, 3-gram
            unique_ngram = [set() for _ in range(0, 3)]
            all_ngram_count = [0 for _ in range(0, 3)]

            split_pred = preprocessing_text(pred).split()
            for n in range(0, 3):
                for i in range(0, len(split_pred) - n):
                    ngram = ' '.join(split_pred[i:i + n + 1])
                    unique_ngram[n].add(ngram)
                    all_ngram_count[n] += 1

            # Sometimes the answer may contain only one word. For 2-gram and 3-gram, the gram count(denominator) may be zero.
            avg_distinct = [len(a) / (b + 1e-6) for a, b in zip(unique_ngram, all_ngram_count)]

            cumulative_distinct.append(statistics.mean(avg_distinct))

    distinct_score["distinct"] = statistics.mean(cumulative_distinct)

    return distinct_score


def bert_score(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Calculate BERTScore Metric

    The BERTScore evaluates the semantic similarity between
    tokens of preds and targets with BERT.
    """
    bert_score = {"bert_score": 0}
    pred_list = []
    target_list = []

    for pred, target in zip(preds, targets):
        pred_list.append(pred)
        target_list.append(target)

    if language == "cn":
        _, _, F = score(pred_list, target_list, lang="zh", verbose=True)
    elif language == "en":
        _, _, F = score(pred_list, target_list, lang="en", verbose=True)

    bert_score["bert_score"] = F.mean().item()

    return bert_score


def calculate_precision_recall_f1(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Precision, Recall and F1-Score Calculation

    The calculation of precision, recall and f1-score is realized by counting
    the number f overlaps between the preds and target. The comparison length
    limited by the shorter one of preds and targets.
    """
    precision_recall_f1 = {"precision": 0, "recall": 0, "f1_score": 0}
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, target in zip(preds, targets):
        if language == "cn":
            pred_list = [char for char in ' '.join(jieba.cut(preprocessing_text(pred))).split()]
            target_list = [char for char in ' '.join(jieba.cut(preprocessing_text(target))).split()]
        elif language == "en":
            pred_list = [char for char in preprocessing_text(pred).split()]
            target_list = [char for char in preprocessing_text(target).split()]

        target_labels = [1] * min(len(target_list), len(pred_list))
        pred_labels = [int(pred_list[i] == target_list[i]) for i in range(0, min(len(target_list), len(pred_list)))]

        precision_scores.append(precision_score(target_labels, pred_labels, zero_division=0))
        recall_scores.append(recall_score(target_labels, pred_labels, zero_division=0))
        f1_scores.append(f1_score(target_labels, pred_labels, zero_division=0))

    precision_recall_f1["precision"] = statistics.mean(precision_scores)
    precision_recall_f1["recall"] = statistics.mean(recall_scores)
    precision_recall_f1["f1_score"] = statistics.mean(f1_scores)

    return precision_recall_f1


def precision(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Calculate Precision Metric

    Calculating precision by counting the number of overlaps between the preds and target.
    """
    precision = {"precision": 0}
    precision["precision"] = calculate_precision_recall_f1(preds, targets, language)["precision"]
    return precision


def recall(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Calculate Recall Metric

    Calculating recall by counting the number of overlaps between the preds and target.
    """
    recall = {"recall": 0}
    recall["recall"] = calculate_precision_recall_f1(preds, targets, language)["recall"]
    return recall


def F1_score(preds: List[str], targets: List[str], language: str) -> Dict[str, float]:
    """Calculate F1-score Metric

    Calculating f1-score by counting the number of overlaps between the preds and target.
    """
    f1 = {"f1_score": 0}
    f1["f1_score"] = calculate_precision_recall_f1(preds, targets, language)["f1_score"]
    return f1
