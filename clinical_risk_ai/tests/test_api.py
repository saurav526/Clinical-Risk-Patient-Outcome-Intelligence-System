def test_import_api():
    from api.main import app
    assert app.title.startswith("Clinical Risk")

# just a simple test to check if the API can be imported and the app object is created successfully.
