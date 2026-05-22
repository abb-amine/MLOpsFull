import os

import ray
import typer
from typing_extensions import Annotated

from madewithml import evaluate, predict, train, tune
from madewithml.analyze import analyze as analyze_data

app = typer.Typer()


@app.callback()
def main(
    github_username: Annotated[str, typer.Option(help="GitHub username for experiment tracking")] = "",
):
    if github_username:
        os.environ["GITHUB_USERNAME"] = github_username
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ.get("GITHUB_USERNAME", "")}})


@app.command()
def run_analyze(
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset")] = "datasets/dataset.csv",
    output_dir: Annotated[str, typer.Option(help="directory to save analysis outputs")] = "outputs",
):
    analyze_data(dataset_loc=dataset_loc, output_dir=output_dir)


@app.command()
def run_train(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment")] = "madewithml",
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset")] = "datasets/dataset.csv",
    train_loop_config: Annotated[
        str, typer.Option(help="training config JSON")
    ] = '{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.5, "lr_patience": 3}',
    num_workers: Annotated[int, typer.Option(help="number of workers")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="CPUs per worker")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="GPUs per worker")] = 0,
    num_samples: Annotated[int, typer.Option(help="number of samples")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs")] = 1,
    batch_size: Annotated[int, typer.Option(help="batch size")] = 256,
    results_fp: Annotated[str, typer.Option(help="path to save results")] = None,
):
    results = train.train_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        train_loop_config=train_loop_config,
        num_workers=num_workers,
        cpu_per_worker=cpu_per_worker,
        gpu_per_worker=gpu_per_worker,
        num_samples=num_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        results_fp=results_fp,
    )
    return results


@app.command()
def run_tune(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment")] = "madewithml",
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset")] = "datasets/dataset.csv",
    initial_params: Annotated[
        str, typer.Option(help="initial hyperparameters JSON")
    ] = '[{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.5, "lr_patience": 3}]',
    num_workers: Annotated[int, typer.Option(help="number of workers")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="CPUs per worker")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="GPUs per worker")] = 0,
    num_runs: Annotated[int, typer.Option(help="number of tuning runs")] = 1,
    num_samples: Annotated[int, typer.Option(help="number of samples")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs")] = 1,
    batch_size: Annotated[int, typer.Option(help="batch size")] = 256,
    results_fp: Annotated[str, typer.Option(help="path to save results")] = None,
):
    results = tune.tune_models(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        initial_params=initial_params,
        num_workers=num_workers,
        cpu_per_worker=cpu_per_worker,
        gpu_per_worker=gpu_per_worker,
        num_runs=num_runs,
        num_samples=num_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        results_fp=results_fp,
    )
    return results


@app.command()
def run_evaluate(
    run_id: Annotated[str, typer.Option(help="run ID to evaluate")],
    dataset_loc: Annotated[str, typer.Option(help="evaluation dataset location")] = "datasets/holdout.csv",
    results_fp: Annotated[str, typer.Option(help="path to save results")] = None,
):
    metrics = evaluate.evaluate(
        run_id=run_id,
        dataset_loc=dataset_loc,
        results_fp=results_fp,
    )
    return metrics


@app.command()
def run_predict(
    run_id: Annotated[str, typer.Option(help="run ID to use for prediction")],
    title: Annotated[str, typer.Option(help="project title")],
    description: Annotated[str, typer.Option(help="project description")] = "",
):
    results = predict.predict(
        run_id=run_id,
        title=title,
        description=description,
    )
    return results


if __name__ == "__main__":
    app()
