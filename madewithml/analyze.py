import warnings
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import typer
from matplotlib.figure import Figure
from wordcloud import STOPWORDS, WordCloud

from madewithml.config import logger
from madewithml.log_utils import log_timing

warnings.filterwarnings("ignore")
sns.set_theme()

app = typer.Typer()


def plot_tag_distribution(df: pd.DataFrame, figsize: tuple = (10, 3)) -> Figure:
    all_tags = Counter(df.tag)
    tags, tag_counts = zip(*all_tags.most_common())
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=list(tags), y=list(tag_counts), ax=ax)
    ax.set_xticklabels(tags, rotation=0, fontsize=12)
    ax.set_title("Tag distribution", fontsize=16)
    ax.set_ylabel("# of projects", fontsize=14)
    return fig


def plot_wordcloud(df: pd.DataFrame, tag: str, figsize: tuple = (10, 3)) -> Figure:
    subset = df[df.tag == tag]
    text = subset.title.values
    cloud = WordCloud(
        stopwords=STOPWORDS,
        background_color="black",
        collocations=False,
        width=500,
        height=300,
    ).generate(" ".join(text))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.imshow(cloud)
    return fig


def find_low_confidence(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_to_index: Dict,
    tag: str,
    threshold: float = 0.5,
) -> List[Dict]:
    index = class_to_index[tag]
    indices = np.where(y_true == index)[0]
    low_confidence = []
    for i in indices:
        prob = y_prob[i][index]
        if prob <= threshold:
            low_confidence.append(
                {
                    "true": tag,
                    "pred": list(class_to_index.keys())[y_prob[i].argmax()],
                    "prob": float(prob),
                }
            )
    return low_confidence


def find_label_issues(y_true: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    from cleanlab.filter import find_label_issues as _find_label_issues

    return _find_label_issues(labels=y_true, pred_probs=pred_probs, return_indices_ranked_by="self_confidence")


def explain_prediction(
    text: str,
    preprocessor,
    predictor,
    class_to_index: Dict,
    num_features: int = 10,
):
    from lime.lime_text import LimeTextExplainer

    def classifier_fn(texts):
        ds = ray.data.from_items([{"title": text, "description": "", "tag": "other"} for text in texts])
        preprocessed_ds = preprocessor.transform(ds)
        outputs = preprocessed_ds.map_batches(predictor.predict_proba)
        y_prob = np.array([d["output"] for d in outputs.take_all()])
        return y_prob

    explainer = LimeTextExplainer(class_names=list(class_to_index.keys()))
    exp = explainer.explain_instance(text, classifier_fn=classifier_fn, top_labels=1, num_features=num_features)
    return exp


def run_behavioral_tests(
    preprocessor,
    predictor,
) -> Dict:
    results = {}
    tests = {
        "invariance": {
            "description": "INVariance via verb injection (changes should not affect outputs)",
            "templates": [
                ("Transformers applied to NLP have {} the ML field.", ["revolutionized", "disrupted"]),
            ],
        },
        "directional": {
            "description": "DIRectional expectations (changes with known outputs)",
            "templates": [
                ("ML applied to {}.", ["text classification", "image classification"]),
            ],
        },
        "min_functionality": {
            "description": "Minimum Functionality Tests (simple input/output pairs)",
            "templates": [
                ("{} is the next big wave in machine learning.", ["natural language processing", "mlops"]),
            ],
        },
    }

    def classifier_fn(texts):
        ds = ray.data.from_items([{"title": text, "description": "", "tag": "other"} for text in texts])
        preprocessed_ds = preprocessor.transform(ds)
        outputs = preprocessed_ds.map_batches(predictor.predict_proba)
        y_prob = np.array([d["output"] for d in outputs.take_all()])
        return y_prob

    for category, config in tests.items():
        category_results = []
        for template, tokens in config["templates"]:
            texts = [template.format(token) for token in tokens]
            y_probs = classifier_fn(texts)
            predictions = [preprocessor.index_to_class[y_prob.argmax()] for y_prob in y_probs]
            category_results.append(
                {
                    "template": template,
                    "inputs": texts,
                    "predictions": predictions,
                }
            )
        results[category] = {
            "description": config["description"],
            "results": category_results,
        }

    return results


@app.command()
@log_timing(logger)
def analyze(
    dataset_loc: str = typer.Option("datasets/dataset.csv", help="location of the dataset"),
    output_dir: str = typer.Option("outputs", help="directory to save analysis outputs"),
):
    import os

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_loc)
    extra = {"samples": len(df), "source": dataset_loc}
    logger.info(f"Loaded {len(df)} samples from {dataset_loc}", extra={"extra_data": extra})

    fig = plot_tag_distribution(df)
    fig.savefig(f"{output_dir}/tag_distribution.png")
    logger.info(f"Saved tag distribution plot to {output_dir}/tag_distribution.png")

    tags_generated = []
    for tag in df.tag.unique():
        fig = plot_wordcloud(df, tag)
        fig.savefig(f"{output_dir}/wordcloud_{tag}.png")
        plt.close(fig)
        tags_generated.append(tag)
        logger.info(f"Saved wordcloud for {tag}")

    logger.info("Analysis complete", extra={"extra_data": {"tags_analyzed": tags_generated, "output_dir": output_dir}})


if __name__ == "__main__":
    app()
