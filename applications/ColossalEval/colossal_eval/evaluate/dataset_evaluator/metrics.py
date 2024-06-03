# Code adapted from https://github.com/THUDM/LongBench/blob/main/metrics.py
# Code adapted from https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# Code adapted from https://github.com/ruixiangcui/AGIEval/blob/main/src/evaluation.py
# https://github.com/SkyworkAI/Skywork/blob/main/eval/eval_gsm8k.py

import difflib
import re
import string
from collections import Counter

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ans_re1 = re.compile(r"(\-?[0-9][0-9\.\,]*)")
ans_re2 = re.compile(r"=\s*(\$?-?[0-9][0-9\.\,]*)")

metrics4subcategory = {
    "pretrain": {
        "perplexity": ["ALL"],
        "ppl_score": ["ALL"],
        "per_byte_perplexity": ["ALL"],
        "per_byte_ppl_score": ["ALL"],
    },
    # The commented are non 4-choice questions.
    "AGIEvalDataset": {
        "combined_single_choice_accuracy": [
            # "lsat-ar",
            # "lsat-lr",
            # "lsat-rc",
            "logiqa-en",
            "sat-math",
            "sat-en",
            # "aqua-rat",
            "sat-en-without-passage",
            "gaokao-english",
            "logiqa-zh",
            "gaokao-chinese",
            "gaokao-geography",
            "gaokao-history",
            "gaokao-biology",
            "gaokao-chemistry",
        ],
        "first_token_accuracy": [
            # "lsat-ar",
            # "lsat-lr",
            # "lsat-rc",
            "logiqa-en",
            "sat-math",
            "sat-en",
            # "aqua-rat",
            "sat-en-without-passage",
            "gaokao-english",
            "logiqa-zh",
            "gaokao-chinese",
            "gaokao-geography",
            "gaokao-history",
            "gaokao-biology",
            "gaokao-chemistry",
        ],
        "single_choice_accuracy": [
            # "lsat-ar",
            # "lsat-lr",
            # "lsat-rc",
            "logiqa-en",
            "sat-math",
            "sat-en",
            # "aqua-rat",
            "sat-en-without-passage",
            "gaokao-english",
            "logiqa-zh",
            "gaokao-chinese",
            "gaokao-geography",
            "gaokao-history",
            "gaokao-biology",
            "gaokao-chemistry",
        ],
        "multi_choice_accuracy": ["jec-qa-kd", "jec-qa-ca", "gaokao-physics", "gaokao-mathqa"],
        "math_equivalence": ["gaokao-mathcloze", "math"],
        "perplexity": ["ALL"],
        "ppl_score_over_choices": [
            "lsat-ar",
            "lsat-lr",
            "lsat-rc",
            "logiqa-en",
            "sat-math",
            "sat-en",
            "aqua-rat",
            "sat-en-without-passage",
            "gaokao-english",
            "logiqa-zh",
            "jec-qa-kd",
            "jec-qa-ca",
            "gaokao-chinese",
            "gaokao-geography",
            "gaokao-history",
            "gaokao-biology",
            "gaokao-chemistry",
            "gaokao-physics",
            "gaokao-mathqa",
        ],
        "ppl_score": ["ALL"],
    },
    "CMMLUDataset": {
        "first_token_accuracy": ["ALL"],
        "single_choice_accuracy": ["ALL"],
        "perplexity": ["ALL"],
        "ppl_score_over_choices": ["ALL"],
        "ppl_score": ["ALL"],
    },
    "GaoKaoBenchDataset": {
        "combined_single_choice_accuracy": [
            "English MCQs",
            "Biology MCQs",
            "Chemistry MCQs",
            "History MCQs",
            "Math I MCQs",
            "Math II MCQs",
            "Political Science MCQs",
        ],
        "first_token_accuracy": [
            "English MCQs",
            "Biology MCQs",
            "Chemistry MCQs",
            "History MCQs",
            "Math I MCQs",
            "Math II MCQs",
            "Political Science MCQs",
        ],
        "single_choice_accuracy": [
            "English MCQs",
            "Biology MCQs",
            "Chemistry MCQs",
            "History MCQs",
            "Math I MCQs",
            "Math II MCQs",
            "Political Science MCQs",
        ],
        "multi_choice_accuracy": [
            "Chinese Lang and Usage MCQs",
            "Chinese Modern Lit",
            "English Fill in Blanks",
            "English Reading Comp",
            "Geography MCQs",
            "Physics MCQs",
            "English Cloze Test",
        ],
        "math_equivalence": ["Math I Fill-in-the-Blank", "Math II Fill-in-the-Blank"],
        "rouge_score": ["English Language Cloze Passage"],
        "rouge_zh_score": [
            "Chinese Language Famous Passages and Sentences Dictation",
            "Chemistry Open-ended Questions",
            "History Open-ended Questions",
            "Biology Open-ended Questions",
            "Political Science Open-ended Questions",
            "English Language Error Correction",
            "Chinese Language Language and Writing Skills Open-ended Questions",
            "Math II Open-ended Questions",
            "Chinese Language Literary Text Reading",
            "Chinese Language Ancient Poetry Reading",
            "Chinese Language Classical Chinese Reading",
            "Physics Open-ended Questions",
            "Math I Open-ended Questions",
            "Geography Open-ended Questions",
            "Chinese Language Practical Text Reading",
        ],
        "perplexity": ["ALL"],
        "ppl_score_over_choices": ["ALL"],
        "ppl_score": ["ALL"],
    },
    "LongBenchDataset": {
        "f1_score": ["hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en", "triviaqa"],
        "f1_zh_score": ["multifieldqa_zh"],
        "rouge_score": ["gov_report", "qmsum", "multi_news", "samsum"],
        "rouge_zh_score": ["dureader", "vcsum"],
        "retrieval_score": ["passage_retrieval_en"],
        "retrieval_zh_score": ["passage_retrieval_zh"],
        "classification_score": ["trec", "lsht"],
        "code_sim_score": ["lcc", "repobench-p"],
        "count_score": ["passage_count"],
        "perplexity": ["ALL"],
        "ppl_score": ["ALL"],
    },
    "MMLUDataset": {
        "first_token_accuracy": ["ALL"],
        "single_choice_accuracy": ["ALL"],
        "accuracy": ["ALL"],
        "perplexity": ["ALL"],
        "ppl_score_over_choices": ["ALL"],
        "ppl_score": ["ALL"],
    },
    "MTBenchDataset": {"mtbench_single_judge": ["ALL"]},
    "CValuesDataset": {"first_token_accuracy": ["ALL"]},
    "SafetyBenchZHDataset": {"first_token_accuracy": ["ALL"]},
    "SafetyBenchENDataset": {"first_token_accuracy": ["ALL"]},
    "GSMDataset": {
        "loss_over_all_tokens": ["ALL"],
        "gsm_accuracy": ["ALL"],
    },
}


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def parse_math_answer(raw_string):
    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left) : -1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(s):
        first_pattern = "\$(.*)\$"
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if "=" in last_match:
                last_match = last_match.split("=")[-1].lstrip(" ")
        return last_match

    def get_answer_without_dollar_sign(s):
        last_match = None
        if "=" in s:
            last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
            if "\\n" in last_match:
                last_match = last_match.split("\\n")[0]
        else:
            pattern = "(?:\\$)?\d+(?:\.\d+)?(?![\w\d])"
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match

    if "\\boxed" in raw_string:
        answer = remove_boxed(last_boxed_only_string(raw_string))
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    return answer


def math_equivalence(prediction, reference, **kwargs):
    prediction = parse_math_answer(prediction)

    if prediction is None and reference is None:
        print("WARNING: Both None")
        return False

    if prediction is None or reference is None:
        return False

    try:
        ss1 = _strip_string(prediction)
        ss2 = _strip_string(reference)
        return ss1 == ss2
    except:
        return prediction == reference


def multi_choice_accuracy(prediction, reference, **kwargs):
    # Only find uppercase letters not surrounded by lowercase letters
    all_classes = kwargs.get("all_classes", None)
    if all_classes:
        pattern = f"(?<![a-z])[{all_classes[0]}-{all_classes[-1]}](?![a-z])"
    else:
        pattern = "(?<![a-z])[A-F](?![a-z])"

    prediction = re.findall(pattern, prediction)
    reference = re.findall(pattern, reference)

    prediction_set = set(prediction)
    reference_set = set(reference)

    score = 0.0
    for p in prediction_set:
        if p not in reference_set:
            return 0.0
        else:
            score += 1 / len(reference_set)

    return score


def accuracy_by_options(question, prediction, reference):
    pattern = r"[A-Z]\. [^\n]+"
    options = re.findall(pattern, question)
    answer = prediction.split("\n\n")[0]

    for option in options:
        choice, content = option.split(". ", 1)

        if choice == reference and content == answer:
            return 1

    return 0


def combined_single_choice_accuracy(prediction, reference, **kwargs):
    return single_choice_accuracy(prediction, reference, **kwargs)


def single_choice_accuracy(prediction, reference, **kwargs):
    # Only find uppercase letters not surrounded by lowercase letters
    all_classes = kwargs.get("all_classes", None)
    if all_classes:
        pattern = f"(?<![a-z])[{all_classes[0]}-{all_classes[-1]}](?![a-z])"
    else:
        pattern = "(?<![a-z])[A-F](?![a-z])"

    prediction = re.findall(pattern, prediction)[0:1]
    reference = re.findall(pattern, reference)

    assert len(reference) == 1

    prediction_set = set(prediction)
    reference_set = set(reference)

    if prediction_set == reference_set:
        return 1.0

    return 0.0


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, reference, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(reference):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, reference, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, reference)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, reference, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, reference)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, reference, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, reference) / 100


def classification_score(prediction, reference, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in reference and match_term != reference:
            em_match_list.remove(match_term)
    if em_match_list != 0:
        if reference in em_match_list:
            score = 1.0 / len(em_match_list)
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == reference)
    return score


def rouge_score(prediction, reference, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [reference], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, reference, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    reference = " ".join(list(jieba.cut(reference, cut_all=False)))
    score = rouge_score(prediction, reference)
    return score


def _f1_score(prediction, reference, **kwargs):
    common = Counter(prediction) & Counter(reference)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(reference)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(prediction, reference, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(reference)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return _f1_score(prediction_tokens, ground_truth_tokens)


def f1_zh_score(prediction, reference, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(reference, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return _f1_score(prediction_tokens, ground_truth_tokens)


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def get_match_str(match, idx):
    match_str = match[idx]
    match_str = match_str.replace(",", "")
    if match_str.endswith("."):
        match_str = match_str[:-1]
    if match_str.endswith(".00"):
        match_str = match_str[:-3]
    if match_str.endswith(".0"):
        match_str = match_str[:-2]
    return match_str


def extract_answer(completion):
    match1 = re.findall(ans_re1, completion)
    match2 = re.findall(ans_re2, completion)
    ans = []
    if match1:
        match_str1 = get_match_str(match1, -1)
        ans.append(match_str1)
    if match2:
        match_str2 = get_match_str(match2, -1).replace("$", "")
        ans.append(match_str2)

    answer = INVALID_ANS
    try:
        if len(ans) > 0:
            answer = eval(ans[-1])
    except Exception as e:
        print(e)
        return answer
    return answer


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    completion = completion.split("answer is")[-1]
    return extract_answer(completion) == gold


def gsm_accuracy(prediction, reference, **kwargs):
    prediction = prediction.split("\n\n\n")[0]
    prediction = prediction.split("\n\n")[0]
    prediction = prediction.split("Question:")[0]

    return 1.0 if is_correct(prediction, reference) else 0.0
