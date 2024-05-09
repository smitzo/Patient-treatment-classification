from classification_lab.projects import iris_feature_study


def test_iris_feature_groups_share_test_population():
    reports = iris_feature_study()
    assert set(reports) == {"sepal", "petal"}
    assert reports["sepal"]["test_rows"] == reports["petal"]["test_rows"]
    assert reports["petal"]["metrics"]["accuracy"] > 0.8


def test_wine_scaling_study_reports_both_models():
    from classification_lab.projects import wine_scaling_study

    reports = wine_scaling_study()
    assert set(reports) == {"scaled", "unscaled"}
    assert reports["scaled"]["classes"] == reports["unscaled"]["classes"]


def test_cancer_study_names_positive_class_explicitly():
    from classification_lab.projects import cancer_detection_study

    result = cancer_detection_study()
    assert result["positive_class"] == "malignant"
    assert 0 <= result["sensitivity"] <= 1
    assert result["false_negatives"] >= 0


def test_digit_confusions_match_offdiagonal_errors():
    from classification_lab.projects import digit_error_study

    result = digit_error_study()
    matrix = result["report"]["metrics"]["confusion_matrix"]
    errors = sum(map(sum, matrix)) - sum(matrix[i][i] for i in range(10))
    assert sum(pair["count"] for pair in result["confusions"]) == errors


def test_churn_contract_support_covers_holdout():
    from classification_lab.projects import churn_contract_study

    result = churn_contract_study()
    assert sum(g["support"] for g in result["breakdown"].values()) == result["report"]["test_rows"]
