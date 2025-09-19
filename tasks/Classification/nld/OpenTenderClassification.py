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
        reference="https://arxiv.org/abs/2509.12340",
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
        bibtex_citation=r"""@misc{banar2025mtebnle5nlembeddingbenchmark,
      title={MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch}, 
      author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      year={2025},
      eprint={2509.12340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12340}, 
}""",
    )

    def dataset_transform(self) -> None:
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"text": f"{ex['title']}\n{ex['description']}"}
            )
