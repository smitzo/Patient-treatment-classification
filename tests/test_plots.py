from classification_lab.plots import confusion_plot


def test_normalized_plot_handles_zero_support_class(tmp_path):
    path = confusion_plot(
        {"classes": ["a", "b"], "confusion_matrix": [[2, 1], [0, 0]]},
        tmp_path / "matrix.png",
        normalize=True,
    )
    assert path.read_bytes().startswith(b"\x89PNG")
