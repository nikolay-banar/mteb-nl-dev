from __future__ import annotations
import mteb
from mteb.overview import TASKS_REGISTRY
from mteb.benchmarks.benchmarks import Benchmark
from mteb.overview import MTEBTasks, get_tasks

from mteb.benchmarks.get_benchmark import BENCHMARK_REGISTRY
from tasks.Retrieval.nld.bBSARDNLRetrieval import bBSARDNLRetrieval
from tasks.Retrieval.nld.DutchNewsArticlesRetrieval import DutchNewsArticlesRetrieval
from tasks.Retrieval.nld.LegalQARetrieval import LegalQANLRetrieval
from tasks.Retrieval.nld.VABBRetrieval import VABBRetrieval
from tasks.Retrieval.nld.OpenTenderRetrieval import OpenTenderRetrieval
from tasks.Retrieval.nld.DBPediaNLv2Retrieval import DBPediaNLv2Retrieval

from tasks.Classification.nld.VaccinChatNLClassification import (
    VaccinChatNLClassification,
)
from tasks.PairClassification.nld.SQuADNLPairClassification import (
    SQuADNLPairClassification,
)
from tasks.Classification.nld.DutchNewsArticlesClassification import (
    DutchNewsArticlesClassification,
)
from tasks.Classification.nld.DutchColaClassification import DutchColaClassification
from tasks.Classification.nld.DutchSarcasticHeadlinesClassification import (
    DutchSarcasticHeadlinesClassification,
)
from tasks.Classification.nld.DutchGovernmentBiasClassification import (
    DutchGovernmentBiasClassification,
)
from tasks.Classification.nld.OpenTenderClassification import OpenTenderClassification
from tasks.Classification.nld.IconclassClassification import IconclassClassification

from tasks.Clustering.nld.DutchNewsArticlesClusteringS2S import (
    DutchNewsArticlesClusteringS2S,
)
from tasks.Clustering.nld.DutchNewsArticlesClusteringP2P import (
    DutchNewsArticlesClusteringP2P,
)
from tasks.Clustering.nld.VABBClusteringS2S import VABBClusteringS2S
from tasks.Clustering.nld.VABBClusteringP2P import VABBClusteringP2P
from tasks.Clustering.nld.OpenTenderClusteringP2P import OpenTenderClusteringP2P
from tasks.Clustering.nld.OpenTenderClusteringS2S import OpenTenderClusteringS2S
from tasks.Clustering.nld.IconclassClusteringS2S import IconclassClusteringS2S


from tasks.PairClassification.nld.SICKNLPairClassification import (
    SICKNLPairClassification,
)
from tasks.PairClassification.nld.XLWICNLPairClassification import (
    XLWICNLPairClassification,
)

from tasks.MultiLabelClassification.nld.CovidDisinformationNLMultiLabelClassification import (
    CovidDisinformationNLMultiLabelClassification,
)
from tasks.MultiLabelClassification.nld.VABBMultiLabelClassification import (
    VABBMultiLabelClassification,
)

from tasks.STS.nld.SICKNLSTS import SICKNLSTS

TASKS_REGISTRY["bBSARDNLRetrieval"] = bBSARDNLRetrieval
TASKS_REGISTRY["DutchNewsArticlesRetrieval"] = DutchNewsArticlesRetrieval
TASKS_REGISTRY["LegalQANLRetrieval"] = LegalQANLRetrieval
TASKS_REGISTRY["VABBRetrieval"] = VABBRetrieval
TASKS_REGISTRY["OpenTenderRetrieval"] = OpenTenderRetrieval
TASKS_REGISTRY["DBPedia-NL-v2"] = DBPediaNLv2Retrieval


TASKS_REGISTRY["VaccinChatNLClassification"] = VaccinChatNLClassification
TASKS_REGISTRY["SQuADNLPairClassification"] = SQuADNLPairClassification
TASKS_REGISTRY["DutchNewsArticlesClassification"] = DutchNewsArticlesClassification
TASKS_REGISTRY["DutchColaClassification"] = DutchColaClassification
TASKS_REGISTRY["DutchSarcasticHeadlinesClassification"] = (
    DutchSarcasticHeadlinesClassification
)
TASKS_REGISTRY["DutchGovernmentBiasClassification"] = DutchGovernmentBiasClassification
TASKS_REGISTRY["OpenTenderClassification"] = OpenTenderClassification
TASKS_REGISTRY["IconclassClassification"] = IconclassClassification

TASKS_REGISTRY["CovidDisinformationNLMultiLabelClassification"] = (
    CovidDisinformationNLMultiLabelClassification
)
TASKS_REGISTRY["VABBMultiLabelClassification"] = VABBMultiLabelClassification

TASKS_REGISTRY["DutchNewsArticlesClusteringS2S"] = DutchNewsArticlesClusteringS2S
TASKS_REGISTRY["DutchNewsArticlesClusteringP2P"] = DutchNewsArticlesClusteringP2P
TASKS_REGISTRY["IconclassClusteringS2S"] = IconclassClusteringS2S


TASKS_REGISTRY["OpenTenderClusteringS2S"] = OpenTenderClusteringS2S
TASKS_REGISTRY["OpenTenderClusteringP2P"] = OpenTenderClusteringP2P

TASKS_REGISTRY["VABBClusteringS2S"] = VABBClusteringS2S
TASKS_REGISTRY["VABBClusteringP2P"] = VABBClusteringP2P

TASKS_REGISTRY["SICK-NL-STS"] = SICKNLSTS

TASKS_REGISTRY["SICKNLPairClassification"] = SICKNLPairClassification
TASKS_REGISTRY["XLWICNLPairClassification"] = XLWICNLPairClassification

MTEB_NL = Benchmark(
    name="MTEB(nl, v1)",
    tasks=MTEBTasks(
        get_tasks(
            languages=["nld"],
            exclusive_language_filter=True,
            tasks=[
                # Classification
                "DutchBookReviewSentimentClassification",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "SIB200Classification",
                "MultiHateClassification",
                "VaccinChatNLClassification",
                "DutchColaClassification",
                "DutchGovernmentBiasClassification",
                "DutchSarcasticHeadlinesClassification",
                "DutchNewsArticlesClassification",
                "OpenTenderClassification",
                "IconclassClassification",
                # PairClassification
                "SICKNLPairClassification",
                "XLWICNLPairClassification",
                # MultiLabelClassification
                "CovidDisinformationNLMultiLabelClassification",
                "MultiEURLEXMultilabelClassification",
                "VABBMultiLabelClassification",
                # Clustering
                "DutchNewsArticlesClusteringS2S",
                "DutchNewsArticlesClusteringP2P",
                "SIB200ClusteringS2S",
                "VABBClusteringS2S",
                "VABBClusteringP2P",
                "OpenTenderClusteringS2S",
                "OpenTenderClusteringP2P",
                "IconclassClusteringS2S",
                # Reranking
                "WikipediaRerankingMultilingual",
                # Retrieval
                "ArguAna-NL",
                "SCIDOCS-NL",
                "SciFact-NL",
                "NFCorpus-NL",
                "BelebeleRetrieval",
                "WebFAQRetrieval",
                "DutchNewsArticlesRetrieval",
                "bBSARDNLRetrieval",
                "LegalQANLRetrieval",
                "OpenTenderRetrieval",
                "VABBRetrieval",
                "WikipediaRetrievalMultilingual",
                # STS
                "SICK-NL-STS",
                "STSBenchmarkMultilingualSTS",
            ],
        )
    ),
    description="MTEB-NL",
    reference="https://arxiv.org/abs/2509.12340",
    citation="""@misc{banar2025mtebnle5nlembeddingbenchmark,
      title={MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch}, 
      author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      year={2025},
      eprint={2509.12340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12340}, 
}""",
    contacts=[],
)

BENCHMARK_REGISTRY["MTEB(nl, v1)"] = MTEB_NL


def get_benchmark(bench_name="MTEB(nl, v1)"):
    return mteb.get_benchmark(bench_name)
