def test_import_api():
    from api.main import app
    assert app.title.startswith("Clinical Risk")
