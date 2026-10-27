def test_package_is_importable() -> None:
    import claim_extractor

    assert claim_extractor.__doc__ is not None
