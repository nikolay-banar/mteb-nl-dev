from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class VABBClusteringS2S(AbsTaskClusteringFast):
    max_fraction_of_documents_to_embed = 1.0
    metadata = TaskMetadata(
        name="VABBClusteringS2S",
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
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="v_measure",
        date=("2009-11-01", "2010-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"title": "sentences"}
            )
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"labels": ex["org_discipline"]}
            )
