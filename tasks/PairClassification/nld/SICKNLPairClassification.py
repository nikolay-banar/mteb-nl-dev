from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskPairClassification import (
    AbsTaskPairClassification,
    PairClassificationDescriptiveStatistics,
)
from collections import Counter


class SICKNLPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICKNLPairClassification",
        dataset={
            "path": "nicolaebanari/mteb-nl-sick",
            "revision": "main",
        },
        description="SICK-NL is a Dutch translation of SICK ",
        reference="https://aclanthology.org/2021.eacl-main.126/",
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="max_ap",
        date=("2020-09-01", "2021-01-01"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""@inproceedings{wijnholds2021sick,
          title={SICK-NL: A Dataset for Dutch Natural Language Inference},
          author={Wijnholds, Gijs and Moortgat, Michael},
          booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
          pages={1474--1479},
          year={2021}
        }""",
        # prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
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
            self.dataset[split] = self.dataset[split].filter(
                lambda ex: ex["label"] != "neutral"
            )
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"labels": 1 if ex["label"] == "entailment" else 0}
            )

            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sentence1"],
                    "sentence2": self.dataset[split]["sentence2"],
                    "labels": self.dataset[split]["labels"],
                }
            ]

        self.dataset = _dataset
