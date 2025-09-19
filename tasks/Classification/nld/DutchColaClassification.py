from __future__ import annotations
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DutchColaClassification(AbsTaskClassification):
    samples_per_label = 128
    metadata = TaskMetadata(
        name="DutchColaClassification",
        description="Dutch CoLA is a corpus of linguistic acceptability for Dutch.",
        reference="https://huggingface.co/datasets/GroNLP/dutch-cola",
        dataset={
            "path": "GroNLP/dutch-cola",
            "revision": "5a4196c41f2b66e02a761d4702f3a7fbee70ef77",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2024-03-01", "2024-05-01"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        domains=["Written"],
        task_subtypes=["Linguistic acceptability"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @misc  {gronlp_2024,
            author       = {Bylinina, Lisa and Abdi, Silvana and Brouwer, Hylke and Elzinga, Martine and Gunput, Shenza and Huisman, Sem and Krooneman, Collin and Poot, David and Top, Jelmer and Weideman, Cain},
            title        = { {Dutch-CoLA (Revision 5a4196c)} },
            year         = 2024,
            url          = { https://huggingface.co/datasets/GroNLP/dutch-cola },
            doi          = { 10.57967/hf/3825 },
            publisher    = { Hugging Face }
            }""",
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"Sentence": "text", "Acceptability": "label"}
            )
