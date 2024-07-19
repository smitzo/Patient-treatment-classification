from classification_lab.datasets import load_dataset
from classification_lab.experiments import train
from classification_lab.html_reports import render_html


def test_html_escapes_labels_and_embedded_json():
    report = train(load_dataset("iris")).report
    report["dataset"] = "<script>alert('x')</script>"
    report["metrics"]["classes"][0] = "<img src=x onerror=evil()>"
    html = render_html(report)
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert "Actual rows; predicted columns" in html
