def test_import():
    try:
        import ttim
    except:
        fail = True
        assert fail is False, "could not import ttim"
    return


if __name__ == "__main__":
    test_import()
