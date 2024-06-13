from classification_lab.datasets import load_dataset
from classification_lab.grouping import group_holdout


def test_entity_groups_never_cross_split():
    data = load_dataset("iris")
    groups = [i // 5 for i in range(len(data.frame))]
    split = group_holdout(data, groups)
    fit_groups = {groups[i] for i in split.X_train.index}
    test_groups = {groups[i] for i in split.X_test.index}
    assert fit_groups.isdisjoint(test_groups)
