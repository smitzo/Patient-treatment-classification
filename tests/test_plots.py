from classification_lab.plots import confusion_plot


def test_normalized_plot_handles_zero_support_class(tmp_path):
    path = confusion_plot(
        {"classes": ["a", "b"], "confusion_matrix": [[2, 1], [0, 0]]},
        tmp_path / "matrix.png",
        normalize=True,
    )
    assert path.read_bytes().startswith(b"\x89PNG")


def test_probability_plot_exports_both_curve_panels(tmp_path):
    from classification_lab.curves import binary_curves
    from classification_lab.plots import probability_curve_plot

    curves = binary_curves(["yes", "no"], [[0.9, 0.1], [0.1, 0.9]], ["yes", "no"], "yes")
    path = probability_curve_plot(curves, tmp_path / "curves.svg")
    assert "Precision-recall" in path.read_text()
    assert "False positive rate" in path.read_text()


def test_importance_plot_keeps_negative_values(tmp_path):
    from classification_lab.plots import importance_plot

    path = importance_plot(
        [
            {"feature": "signal", "mean": 0.2, "std": 0.01},
            {"feature": "noise", "mean": -0.05, "std": 0.02},
        ],
        tmp_path / "importance.svg",
    )
    assert "noise" in path.read_text()
    assert "Permutation importance" in path.read_text()
