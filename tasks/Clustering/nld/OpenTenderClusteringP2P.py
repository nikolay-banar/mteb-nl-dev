from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    clustering_downsample,
)
import datasets


class OpenTenderClusteringP2P(AbsTaskClusteringFast):
    max_fraction_of_documents_to_embed = 1.0
    metadata = TaskMetadata(
        name="OpenTenderClusteringP2P",
        dataset={
            "path": "clips/mteb-nl-opentender-cls",
            "revision": "53221b9d10649a531dceccdab8155ab795a59bbb",
        },
        description="This dataset contains all the articles published by the NOS as of the 1st of January 2010. The "
        "data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news "
        "organizations in the Netherlands.",
        reference="https://arxiv.org/abs/2509.12340",
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="v_measure",
        date=("2025-08-01", "2025-08-10"),
        domains=["Government", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{banar2025mtebnle5nlembeddingbenchmark,
      title={MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch}, 
      author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      year={2025},
      eprint={2509.12340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12340}, 
}""",
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {
                    "labels": ex["label"],
                    "sentences": f"{ex['title']}\n{ex['description']}",
                }
            )
