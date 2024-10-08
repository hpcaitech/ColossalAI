import copy
import csv
import os
from typing import Dict, List

from colossalai.logging import DistributedLogger

from .base import BaseDataset

ceval_subject_mapping = {
    "computer_network": ["Computer Network", "计算机网络", "STEM"],
    "operating_system": ["Operating System", "操作系统", "STEM"],
    "computer_architecture": ["Computer Architecture", "计算机组成", "STEM"],
    "college_programming": ["College Programming", "大学编程", "STEM"],
    "college_physics": ["College Physics", "大学物理", "STEM"],
    "college_chemistry": ["College Chemistry", "大学化学", "STEM"],
    "advanced_mathematics": ["Advanced Mathematics", "高等数学", "STEM"],
    "probability_and_statistics": ["Probability and Statistics", "概率统计", "STEM"],
    "discrete_mathematics": ["Discrete Mathematics", "离散数学", "STEM"],
    "electrical_engineer": ["Electrical Engineer", "注册电气工程师", "STEM"],
    "metrology_engineer": ["Metrology Engineer", "注册计量师", "STEM"],
    "high_school_mathematics": ["High School Mathematics", "高中数学", "STEM"],
    "high_school_physics": ["High School Physics", "高中物理", "STEM"],
    "high_school_chemistry": ["High School Chemistry", "高中化学", "STEM"],
    "high_school_biology": ["High School Biology", "高中生物", "STEM"],
    "middle_school_mathematics": ["Middle School Mathematics", "初中数学", "STEM"],
    "middle_school_biology": ["Middle School Biology", "初中生物", "STEM"],
    "middle_school_physics": ["Middle School Physics", "初中物理", "STEM"],
    "middle_school_chemistry": ["Middle School Chemistry", "初中化学", "STEM"],
    "veterinary_medicine": ["Veterinary Medicine", "兽医学", "STEM"],
    "college_economics": ["College Economics", "大学经济学", "Social Science"],
    "business_administration": ["Business Administration", "工商管理", "Social Science"],
    "marxism": ["Marxism", "马克思主义基本原理", "Social Science"],
    "mao_zedong_thought": ["Mao Zedong Thought", "毛泽东思想和中国特色社会主义理论体系概论", "Social Science"],
    "education_science": ["Education Science", "教育学", "Social Science"],
    "teacher_qualification": ["Teacher Qualification", "教师资格", "Social Science"],
    "high_school_politics": ["High School Politics", "高中政治", "Social Science"],
    "high_school_geography": ["High School Geography", "高中地理", "Social Science"],
    "middle_school_politics": ["Middle School Politics", "初中政治", "Social Science"],
    "middle_school_geography": ["Middle School Geography", "初中地理", "Social Science"],
    "modern_chinese_history": ["Modern Chinese History", "近代史纲要", "Humanities"],
    "ideological_and_moral_cultivation": ["Ideological and Moral Cultivation", "思想道德修养与法律基础", "Humanities"],
    "logic": ["Logic", "逻辑学", "Humanities"],
    "law": ["Law", "法学", "Humanities"],
    "chinese_language_and_literature": ["Chinese Language and Literature", "中国语言文学", "Humanities"],
    "art_studies": ["Art Studies", "艺术学", "Humanities"],
    "professional_tour_guide": ["Professional Tour Guide", "导游资格", "Humanities"],
    "legal_professional": ["Legal Professional", "法律职业资格", "Humanities"],
    "high_school_chinese": ["High School Chinese", "高中语文", "Humanities"],
    "high_school_history": ["High School History", "高中历史", "Humanities"],
    "middle_school_history": ["Middle School History", "初中历史", "Humanities"],
    "civil_servant": ["Civil Servant", "公务员", "Other"],
    "sports_science": ["Sports Science", "体育学", "Other"],
    "plant_protection": ["Plant Protection", "植物保护", "Other"],
    "basic_medicine": ["Basic Medicine", "基础医学", "Other"],
    "clinical_medicine": ["Clinical Medicine", "临床医学", "Other"],
    "urban_and_rural_planner": ["Urban and Rural Planner", "注册城乡规划师", "Other"],
    "accountant": ["Accountant", "注册会计师", "Other"],
    "fire_engineer": ["Fire Engineer", "注册消防工程师", "Other"],
    "environmental_impact_assessment_engineer": [
        "Environmental Impact Assessment Engineer",
        "环境影响评价工程师",
        "Other",
    ],
    "tax_accountant": ["Tax Accountant", "税务师", "Other"],
    "physician": ["Physician", "医师资格", "Other"],
}

default_inference_kwargs = {
    "calculate_loss": False,
    "all_classes": ["A", "B", "C", "D"],
    "language": "Chinese",
    "calculate_overall_loss": False,
    "max_new_tokens": 32,
}


def get_few_shot_data(data: List[Dict], subject):
    few_shot_data = [f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"]
    for i in data:
        few_shot_data.append(i["input"] + i["target"])
    return few_shot_data


class CEvalDataset(BaseDataset):
    """
    Dataset class for CEval dataset.
    Data source: https://huggingface.co/datasets/ceval/ceval-exam
    This dataset class will convert the original dataset into the inference dataset.
    """

    @staticmethod
    def load(path: str, logger: DistributedLogger, few_shot: bool, *args, **kwargs) -> List[Dict]:
        dataset = {"dev": {}, "test": {}}
        for split in ["dev", "test"]:
            files = os.listdir(os.path.join(path, split))
            files.sort()

            for file in files:
                subject = file[0 : -len(f"_{split}.csv")]
                subject = ceval_subject_mapping[subject][1]

                file_dir = os.path.join(path, split, file)

                dataset[split][subject] = {"data": []}

                # It's been tested that each data sample in one subcategory have same inference arguments.
                dataset[split][subject]["inference_kwargs"] = copy.deepcopy(default_inference_kwargs)

                if split == "test" and few_shot:
                    dataset[split][subject]["inference_kwargs"]["few_shot_data"] = get_few_shot_data(
                        dataset["dev"][subject]["data"], subject
                    )

                with open(file_dir, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    _ = next(reader)
                    for row in reader:
                        # Dev split have answer and explanation so len(row) is 8
                        # But test split doesn't contain answer and explanation, so len(row) is 6
                        assert len(row) >= 6
                        choices = f"A. {row[2]}\nB. {row[3]}\nC. {row[4]}\nD. {row[5]}"
                        data_sample = {
                            "dataset": "ceval",
                            "split": split,
                            "category": subject,
                            "instruction": f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。",
                            "input": f"题目：{row[1]}\n{choices}\n答案：",
                            "output": "",
                            "target": row[6] if split == "dev" else "",
                            "id": int(row[0]),
                        }

                        dataset[split][subject]["data"].append(data_sample)

        return dataset
