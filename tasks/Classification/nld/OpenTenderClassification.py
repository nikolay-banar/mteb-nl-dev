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
        description="This dataset contains Belgian and Dutch tender calls from OpenTender in Dutch",
        reference="https://aclanthology.org/2021.findings-emnlp.56.pdf",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        date=("2025-08-01", "2025-08-10"),
        domains=["Government", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""""",
    )

    def dataset_transform(self) -> None:
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"text": f"{ex['title']}\n{ex['description']}"}
            )
