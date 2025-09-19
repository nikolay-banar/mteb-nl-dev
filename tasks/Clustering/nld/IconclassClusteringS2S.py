from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class IconclassClusteringS2S(AbsTaskClusteringFast):
    max_fraction_of_documents_to_embed = 1.0
    metadata = TaskMetadata(
        name="IconclassClusteringS2S",
        dataset={
            "path": "clips/mteb-nl-iconclass-cls",
            "revision": "1cd02f1579dab39fedc95de8cc15fd620557a9f2",
        },
        description="Iconclass is an iconographic thesaurus, which is widely used in the digital heritage domain to "
        "describe subjects depicted in artworks. The task is to classify the first layer of Iconclass",
        reference="https://dl.acm.org/doi/pdf/10.1145/3575865",
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="v_measure",
        date=("2009-11-01", "2010-01-01"),
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{banar2023transfer,
          title={Transfer learning for the visual arts: The multi-modal retrieval of iconclass codes},
          author={Banar, Nikolay and Daelemans, Walter and Kestemont, Mike},
          journal={ACM Journal on Computing and Cultural Heritage},
          volume={16},
          number={2},
          pages={1--16},
          year={2023},
          publisher={ACM New York, NY}
}""",
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"labels": ex["label"], "sentences": ex["text"]}
            )
