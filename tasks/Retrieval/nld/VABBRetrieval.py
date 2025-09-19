from __future__ import annotations
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class VABBRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VABBRetrieval",
        description="This dataset contains all the articles published by the NOS as of the 1st of January 2010. The "
        "data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news "
        "organizations in the Netherlands.",
        reference="https://zenodo.org/records/14214806",
        dataset={
            "path": "clips/mteb-nl-vabb-ret",
            "revision": "af4a1e5b3ed451103894f86ff6b3ce85085d7b48",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2009-11-01", "2010-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
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
