from classification_lab.subgroups import subgroup_metrics


def test_small_groups_suppressed_and_missing_members_counted():
    report = subgroup_metrics(
        ["a", "b", "a", "b"], ["a"] * 4, ["large", "large", None, "tiny"], ["a", "b"], min_support=2
    )
    assert report["large"]["support"] == 2
    assert report["large"]["metrics"]["accuracy"] == 0.5
    assert report["tiny"]["metrics"] is None
    assert report["<missing>"]["support"] == 1
