from __future__ import annotations
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class VABBClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VABBClassification",
        dataset={
            "path": "clips/mteb-nl-vabb-cls",
            "revision": "544acc2e46909eab2b49962b043a18b9c9772770",
        },
        description="This dataset contains the fourteenth edition of the Flemish Academic Bibliography for the Social "
        "Sciences and Humanities (VABB-SHW), a database of academic publications from the social sciences "
        "and humanities authored by researchers affiliated to Flemish universities (more information). "
        "Publications in the database are used as one of the parameters of the Flemish performance-based "
        "research funding system",
        reference="https://zenodo.org/records/14214806",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        date=("2020-01-01", "2021-04-01"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@dataset{aspeslagh2024vabb,
              author       = {Aspeslagh, Pieter and Guns, Raf and Engels, Tim C. E.},
              title        = {VABB-SHW: Dataset of Flemish Academic Bibliography for the Social Sciences and Humanities (edition 14)},
              year         = {2024},
              publisher    = {Zenodo},
              doi          = {10.5281/zenodo.14214806},
              url          = {https://doi.org/10.5281/zenodo.14214806}
            }
            """,
    )

    def dataset_transform(self) -> None:
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"org_discipline": "label"}
            )
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"text": f"{ex['title']}\n{ex['abstract']}"}
            )
