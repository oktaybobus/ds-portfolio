"""The ``dsj`` command-line interface.

One entry point for the whole portfolio: list what exists, inspect a dataset,
train, benchmark, predict, and launch a project's Streamlit demo - so a reviewer
can go from ``git clone`` to a scored model with two commands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from dsjourney import eda, serving
from dsjourney.artifacts import bundle_exists, load_bundle
from dsjourney.benchmark import compare_models, compare_text_models
from dsjourney.config import load_project_config
from dsjourney.paths import available_projects, project_dir
from dsjourney.pipeline import load_pipeline
from dsjourney.preprocess import split_and_scale
from dsjourney.training import train_clustering, train_supervised

app = typer.Typer(
    name="dsj",
    help="Train, evaluate and serve the projects in this data-science portfolio.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ProjectArg = Annotated[str, typer.Argument(help="Project name, as listed by 'dsj list'")]


@app.command("list")
def list_projects() -> None:
    """List every project with its task and whether it has been trained."""
    names = available_projects()
    if not names:
        console.print("[yellow]No projects found under projects/[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Portfolio projects", header_style="bold cyan")
    for column in ("project", "task", "title", "trained", "headline metric"):
        table.add_column(column)

    for name in names:
        config = load_project_config(name)
        trained = bundle_exists(name)
        metric = "-"
        if trained:
            metrics = load_bundle(name).metrics
            metric = _headline_metric(metrics)
        table.add_row(
            name,
            config.task,
            config.title,
            "[green]yes[/green]" if trained else "[dim]no[/dim]",
            metric,
        )
    console.print(table)


@app.command()
def info(project: ProjectArg) -> None:
    """Show a project's configuration and, if present, its saved metrics."""
    config = load_project_config(project)
    console.print_json(config.model_dump_json(indent=2))
    if bundle_exists(project):
        bundle = load_bundle(project)
        console.print("\n[bold]Saved model[/bold]")
        console.print_json(json.dumps({"metrics": bundle.metrics, **bundle.extra}, default=str))
    else:
        console.print(f"\n[dim]Not trained yet. Run: dsj train {project}[/dim]")


@app.command()
def eda_report(
    project: ProjectArg,
    rows: Annotated[int, typer.Option(help="How many overview rows to print")] = 40,
) -> None:
    """Print a column overview and missing-value report for a project's raw data."""
    module = load_pipeline(project)
    frame = module.load_raw()
    console.print(f"[bold]{project}[/bold]: {frame.shape[0]:,} rows x {frame.shape[1]} columns\n")
    console.print(eda.overview(frame).head(rows).to_string())
    missing = eda.missing_report(frame)
    console.print("\n[bold]Missing values[/bold]")
    console.print(missing.to_string() if len(missing) else "[green]none[/green]")


@app.command()
def train(
    project: ProjectArg,
    benchmark: Annotated[
        bool, typer.Option(help="Sweep every estimator and keep the winner")
    ] = False,
    save: Annotated[bool, typer.Option(help="Write artifacts to artifacts/<project>/")] = True,
) -> None:
    """Train a project end to end and save the model bundle."""
    config = load_project_config(project)
    module = load_pipeline(project)

    console.print(f"[cyan]Loading data for {project}...[/cyan]")
    features = module.build_features(module.load_raw())

    console.print(
        f"[cyan]Training ({'benchmark sweep' if benchmark else config.model.estimator})...[/cyan]"
    )
    report = (
        train_clustering(config, features, save=save)
        if config.task == "clustering"
        else train_supervised(
            config,
            features,
            benchmark=benchmark,
            save=save,
            inverse_transform=getattr(module, "postprocess", None),
        )
    )

    console.print(f"[green]{report.summary()}[/green]")
    if report.benchmark is not None:
        console.print(report.benchmark.table.head(10).to_string(index=False))
    if report.artifacts_dir:
        console.print(f"[dim]Artifacts: {report.artifacts_dir}[/dim]")


@app.command()
def benchmark(
    project: ProjectArg,
    rank_by: Annotated[str | None, typer.Option(help="Metric to rank on")] = None,
) -> None:
    """Compare every registered estimator on a project without saving anything."""
    config = load_project_config(project)
    if config.task == "clustering" or config.target is None:
        console.print("[red]benchmark supports supervised projects only[/red]")
        raise typer.Exit(code=2)

    module = load_pipeline(project)
    features = module.build_features(module.load_raw())

    if config.task == "text-classification":
        result = _benchmark_text(config, module, features, rank_by)
    else:
        split = split_and_scale(
            features, config.target, config.split, scale_columns=config.model.scale_features
        )
        result = compare_models(
            config.task, split.x_train, split.y_train, split.x_test, split.y_test, rank_by=rank_by
        )
    console.print(result.table.to_string(index=False))
    console.print(f"\n[green]Winner: {result.best_name}[/green]")


@app.command()
def predict(
    project: ProjectArg,
    payload: Annotated[
        str, typer.Option("--json", help="Inline JSON record, or @path/to/file.json")
    ],
) -> None:
    """Score a single record against a project's saved model."""
    record = _read_payload(payload)
    module = load_pipeline(project)
    bundle = load_bundle(project)

    row = bundle.prepare(module.prepare_input(record))
    raw_prediction = bundle.model.predict(row)
    postprocess = getattr(module, "postprocess", None)
    value = postprocess(raw_prediction) if callable(postprocess) else raw_prediction

    console.print_json(
        json.dumps({"project": project, "prediction": _to_jsonable(value)}, default=str)
    )


@app.command()
def serve(project: ProjectArg) -> None:
    """Launch a project's Streamlit demo."""
    script = project_dir(project) / "app.py"
    if not script.is_file():
        console.print(f"[red]No Streamlit app at {script}[/red]")
        raise typer.Exit(code=2)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)], check=False)


def _benchmark_text(config: Any, module: Any, features: Any, rank_by: str | None) -> Any:
    """Split a text project's documents and sweep TF-IDF pipelines over them."""
    from sklearn.model_selection import train_test_split

    documents = features[module.TEXT_COLUMN]
    labels = features[config.target]
    x_train, x_test, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=labels if config.split.stratify else None,
    )
    return compare_text_models(x_train, y_train, x_test, y_test, rank_by=rank_by or "f1")


@app.command()
def api(
    host: Annotated[str, typer.Option(help="Interface to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on")] = 8000,
    reload: Annotated[bool, typer.Option(help="Restart on source changes")] = False,
) -> None:
    """Serve every trained project over HTTP; docs at /docs."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]FastAPI is not installed. Run: uv sync --extra api[/red]")
        raise typer.Exit(code=2) from None

    ready = [p.name for p in serving.servable_projects() if p.servable]
    console.print(f"[green]Serving {len(ready)} project(s):[/green] {', '.join(ready) or 'none'}")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/dim]")
    uvicorn.run("service.app:app", host=host, port=port, reload=reload)


def _headline_metric(metrics: dict[str, float]) -> str:
    """Pick the single metric worth showing in the project list."""
    for key in ("r2", "f1", "accuracy", "silhouette"):
        if key in metrics:
            return f"{key}={metrics[key]:.4f}"
    return "-"


def _read_payload(payload: str) -> dict[str, Any]:
    """Parse a record given inline as JSON or as ``@file.json``."""
    text = Path(payload[1:]).read_text(encoding="utf-8") if payload.startswith("@") else payload
    record = json.loads(text)
    if not isinstance(record, dict):
        raise typer.BadParameter("payload must be a JSON object")
    return record


def _to_jsonable(value: Any) -> Any:
    """Reduce numpy scalars and single-element arrays to plain Python values."""
    if hasattr(value, "tolist"):
        listed = value.tolist()
        return listed[0] if isinstance(listed, list) and len(listed) == 1 else listed
    return value


if __name__ == "__main__":  # pragma: no cover
    app()
