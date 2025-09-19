from __future__ import annotations
import mteb
import argparse
from add_datasets import get_benchmark
from add_models import get_model


def main(
    model_name,
    model_type,
    task_name,
    batch_size,
    output_folder,
    model_revision=None,
    max_seq_length=None,
    use_custom_prompts=False,
):
    model, encode_kwargs = get_model(
        model_name=model_name,
        model_type=model_type,
        batch_size=batch_size,
        model_revision=model_revision,
        max_seq_length=max_seq_length,
        use_custom_prompts=use_custom_prompts,
    )

    if task_name == "MTEB(nl, v1)":
        tasks = get_benchmark(bench_name=task_name)
    else:
        tasks = mteb.get_tasks(tasks=[task_name], languages=["nld"])

    evaluation = mteb.MTEB(tasks)

    evaluation.run(model, output_folder=output_folder, encode_kwargs=encode_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation with a given model and task set"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="HuggingFace model identifier (or MTEB alias)",
    )

    parser.add_argument(
        "--model_revision", type=str, default=None, help="HuggingFace model revision"
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="MTEB(nl, v1)",
        help="HuggingFace model identifier (or MTEB alias)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )

    parser.add_argument(
        "--max_seq_length", type=int, default=None, help="max_seq_length for inference"
    )

    parser.add_argument(
        "--model_type",
        choices=[
            "e5",
            "cls",
            "llm",
            "e5-inst",
            "mean",
            None,
        ],  # restricts allowed values
        default=None,
        help="Model types",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Where to write evaluation results; defaults to results/<model_name>",
    )

    parser.add_argument(
        "--use_custom_prompts",
        action="store_true",
        help="Enable custom prompts instead of defaults",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        model_type=args.model_type,
        task_name=args.task_name,
        batch_size=args.batch_size,
        output_folder=args.output_folder,
        model_revision=args.model_revision,
        max_seq_length=args.max_seq_length,
        use_custom_prompts=args.use_custom_prompts,
    )
