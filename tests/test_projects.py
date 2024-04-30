from classification_lab.projects import iris_feature_study


def test_iris_feature_groups_share_test_population():
    reports = iris_feature_study()
    assert set(reports) == {"sepal", "petal"}
    assert reports["sepal"]["test_rows"] == reports["petal"]["test_rows"]
    assert reports["petal"]["metrics"]["accuracy"] > 0.8
