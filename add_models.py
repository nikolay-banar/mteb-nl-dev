import mteb
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from mteb.models import e5_models
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper
from functools import partial
from mteb.models.overview import MODEL_REGISTRY

PROMPTS = {
    #     # Classification
    "DutchBookReviewSentimentClassification": "Classify the given book review into positive or negative sentiment",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "SIB200Classification": "Classify the given passage into the appropriate topic or theme",
    "MultiHateClassification": "Classify the given text as either hateful or not hateful",
    "VaccinChatNLClassification": "Given a user utterance as query, find the user intents",
    "DutchColaClassification": "Classify the given sentence as either grammatically acceptable or not acceptable",
    "DutchGovernmentBiasClassification": "Classify the given news article headline as either biased or not biased",
    "DutchSarcasticHeadlinesClassification": "Classify the given news article headline as either sarcastic or not "
    "sarcastic",
    "DutchNewsArticlesClassification": "Classify the given news article into the appropriate topic or theme",
    "OpenTenderClassification": "Classify the given tender description into the appropriate topic or theme",
    "IconclassClassification": "Classify the given artwork title into the appropriate topic or theme",
    # # PairClassification
    "SICKNLPairClassification": "Retrieve text that are semantically similar to the given text",
    "SQuADNLPairClassification": "Retrieve text that are semantically similar to the given text",
    "XLWICNLPairClassification": "Retrieve text that are semantically similar to the given text",
    # # MultiLabelClassification
    "CovidDisinformationNLMultiLabelClassification": "Classify COVID-19-related social media posts into all "
    "applicable disinformation categories",
    "MultiEURLEXMultilabelClassification": "Classify the topics of a legal document",
    "VABBMultiLabelClassification": "Classify the topics of a scientific paper based on the abstract",
    # Clustering
    "DutchNewsArticlesClusteringS2S": "Identify the main category of news articles based on the titles",
    "DutchNewsArticlesClusteringP2P": "Identify the main category of news articles based on the titles and content",
    "SIB200ClusteringS2S": "Identify the topic or theme of passages",
    "VABBClusteringS2S": "Identify the main category of scientific papers based on the titles",
    "VABBClusteringP2P": "Identify the main category of scientific papers based on the titles and abstracts",
    "OpenTenderClusteringS2S": "Identify the main category of tenders based on the titles",
    "OpenTenderClusteringP2P": "Identify the main category of tenders based on the titles and descriptions",
    "IconclassClusteringS2S": "Identify the topic or theme of artworks based on the titles",
    # Reranking
    "WikipediaRerankingMultilingual": "Given a question, retrieve Wikipedia passages that answer the question",
    # Retrieval
    "BelebeleRetrieval": "Given a query, retrieve the relevant passages",
    "WebFAQRetrieval": "Given a question, retrieve replies that best answer the question",
    "DutchNewsArticlesRetrieval": "Given a title, retrieve a news article that best fits the title",
    "bBSARDNLRetrieval": "Given a legal question, retrieve documents that can help answer the question",
    "LegalQANLRetrieval": "Given a legal question, retrieve documents that can help answer the question",
    "OpenTenderRetrieval": "Given a title, retrieve a description of a tender that best fits the title",
    "VABBRetrieval": "Given a title, retrieve a scientific abstract that best fits the title",
    "WikipediaRetrievalMultilingual": "Given a question, retrieve Wikipedia passages that answer the question",
    "ArguAna-NL": "Given a claim, find documents that refute the claim",
    "SCIDOCS-NL": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact-NL": "Given a scientific claim, retrieve documents that support or refute the claim",
    "NFCorpus-NL": "Given a question, retrieve relevant documents that best answer the question",
    # STS
    "SICK-NL-STS": "Retrieve semantically similar text",
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text",
}


def get_model(
    model_name,
    model_type,
    model_revision,
    batch_size,
    max_seq_length,
    use_custom_prompts,
):
    model_kwargs = {}

    if model_revision:
        model_kwargs["revision"] = model_revision
    encode_kwargs = {"batch_size": batch_size}

    if model_type == "cls" or model_type == "mean":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["trust_remote_code"] = True
        # "trust_remote_code": True

        transformer = Transformer(
            model_name,
            model_args=model_kwargs,
            tokenizer_args={"revision": model_revision},
            config_args={"revision": model_revision},
        )

        pooling = Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode="cls" if model_type == "cls" else "mean",
        )

        sent_transformer = SentenceTransformer(modules=[transformer, pooling])
        sent_transformer.model_card_data.base_model_revision = model_revision
        sent_transformer.model_card_data.model_name = model_name

        if max_seq_length:
            sent_transformer.max_seq_length = max_seq_length

        m_type = ModelMeta(
            loader=partial(
                SentenceTransformerWrapper,
                model=sent_transformer,
                revision=model_revision,
            ),
            name=model_name,
            languages=["nld-Latn"],
            open_weights=True,
            revision=model_revision,
            release_date=None,
            n_parameters=None,
            memory_usage_mb=None,
            embed_dim=None,
            license=None,
            max_tokens=None,
            reference=None,
            similarity_fn_name="cosine",
            framework=["Sentence Transformers", "PyTorch"],
            use_instructions=False,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
            adapted_from=None,
        )

        MODEL_REGISTRY[m_type.name] = m_type
        encode_kwargs = {"normalize_embeddings": True}

    elif model_type == "e5":
        model_kwargs["torch_dtype"] = torch.float16
        sent_transformer = SentenceTransformer(
            model_name, revision=model_revision, model_kwargs=model_kwargs
        )

        if max_seq_length:
            sent_transformer.max_seq_length = max_seq_length

        e5_type = ModelMeta(
            loader=partial(
                SentenceTransformerWrapper,
                model=sent_transformer,
                revision=model_revision,
                model_prompts=e5_models.model_prompts,
            ),
            name=model_name,
            languages=["nld-Latn"],
            open_weights=True,
            revision=model_revision,
            release_date=None,
            n_parameters=None,
            memory_usage_mb=None,
            embed_dim=None,
            license=None,
            max_tokens=None,
            reference=None,
            similarity_fn_name="cosine",
            framework=["Sentence Transformers", "PyTorch"],
            use_instructions=True,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
            adapted_from=None,
        )

        MODEL_REGISTRY[e5_type.name] = e5_type
        encode_kwargs = {"normalize_embeddings": True}

    elif model_type == "e5-inst":
        model_kwargs["torch_dtype"] = torch.float16

        e5_instruct = ModelMeta(
            loader=partial(  # type: ignore
                InstructSentenceTransformerWrapper,
                model_name=model_name,
                instruction_template="Instruct: {instruction}\nQuery: ",
                revision=model_revision,
                apply_instruction_to_passages=False,
                prompts_dict=PROMPTS,
                model_kwargs=model_kwargs,
            ),
            name=model_name,
            languages=None,
            open_weights=True,
            revision=model_revision,
            release_date=None,
            framework=[],
            similarity_fn_name="cosine",
            use_instructions=True,
            reference=None,
            n_parameters=None,
            memory_usage_mb=None,
            embed_dim=None,
            license=None,
            max_tokens=None,
            adapted_from=None,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
        )
        encode_kwargs = {"normalize_embeddings": True}

        MODEL_REGISTRY[e5_instruct.name] = e5_instruct

    model = mteb.get_model(
        model_name=model_name,
        revision=model_revision,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )

    if use_custom_prompts:
        model.prompts_dict = PROMPTS

    return model, encode_kwargs
