from __future__ import annotations
from mteb.abstasks.AbsTaskPairClassification import (
    AbsTaskPairClassification,
    PairClassificationDescriptiveStatistics,
)
from mteb.abstasks.TaskMetadata import TaskMetadata
from collections import Counter


class XLWICNLPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XLWICNLPairClassification",
        description="The Word-in-Context dataset (WiC) addresses the dependence on sense inventories by reformulating "
        "the standard disambiguation task as a binary classification problem; but, it is limited to the "
        "English language. We put forward a large multilingual benchmark, XL-WiC, featuring gold standards "
        "in 12 new languages from varied language families and with different degrees of resource "
        "availability, opening room for evaluation scenarios such as zero-shot cross-lingual transfer. ",
        reference="https://aclanthology.org/2020.emnlp-main.584.pdf",
        dataset={
            "path": "pasinit/xlwic",
            "revision": "main",
            "name": "xlwic_en_nl",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        date=("2019-10-04", "2019-10-04"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="max_ap",
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{raganato2020xl,
          title={XL-WiC: A multilingual benchmark for evaluating semantic contextualization},
          author={Raganato, A and Pasini, T and Camacho-Collados, J and Pilehvar, M and others},
          booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
          pages={7193--7206},
          year={2020},
          organization={Association for Computational Linguistics (ACL)}
        }""",
    )

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> PairClassificationDescriptiveStatistics:
        dataset = self.dataset[split][0]

        sentence1 = (
            dataset["sentence1"][0]
            if len(dataset["sentence1"]) == 1
            else dataset["sentence1"]
        )
        sentence2 = (
            dataset["sentence2"][0]
            if len(dataset["sentence2"]) == 1
            else dataset["sentence2"]
        )
        labels = (
            dataset["labels"][0] if len(dataset["labels"]) == 1 else dataset["labels"]
        )

        sentence1_len = [len(sentence) for sentence in sentence1]
        total_sentence1_len = sum(sentence1_len)
        sentence2_len = [len(sentence) for sentence in sentence2]
        total_sentence2_len = sum(sentence2_len)
        label_count = Counter(labels)
        return PairClassificationDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=total_sentence1_len + total_sentence2_len,
            min_sentence1_length=min(sentence1_len),
            avg_sentence1_length=total_sentence1_len / len(sentence1),
            max_sentence1_length=max(sentence1_len),
            unique_sentence1=len(set(sentence1)),
            min_sentence2_length=min(sentence2_len),
            avg_sentence2_length=total_sentence2_len / len(sentence2),
            max_sentence2_length=max(sentence2_len),
            unique_sentence2=len(set(sentence2)),
            unique_labels=len(set(labels)),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )

    def dataset_transform(self) -> None:
        _dataset = {}

        for split in self.dataset:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["context_1"],
                    "sentence2": self.dataset[split]["context_2"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset
