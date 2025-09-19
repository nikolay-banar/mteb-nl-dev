from __future__ import annotations
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OpenTenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OpenTenderClassification",
        dataset={
            "path": "clips/mteb-nl-opentender-cls",
            "revision": "53221b9d10649a531dceccdab8155ab795a59bbb",
        },
        description="The dataset is curated to address questions of interest to journalists, fact-checkers, "
        "social media platforms, policymakers, and the general public.",
        reference="https://aclanthology.org/2021.findings-emnlp.56.pdf",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        date=("2020-01-01", "2021-04-01"),
        domains=["Web", "Social", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""""",
        # prompt="Given a sentence as query, find sensitive topics",
    )

    def dataset_transform(self) -> None:
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"text": f"{ex['title']}\n{ex['description']}"}
            )
