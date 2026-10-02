from pathlib import Path
from src.data.validation import load_and_validate

def test_dataset_shape():
    path = Path("data/raw/clinical_risk_5000.csv")
    df = load_and_validate(path)
    assert len(df) == 5000
    assert "readmitted_30d" in df.columns
