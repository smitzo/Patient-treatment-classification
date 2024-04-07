def test_package_has_version():
    from classification_lab import __version__

    assert __version__ == "0.1.0"
