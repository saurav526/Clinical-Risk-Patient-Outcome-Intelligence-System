from agent.tools import patient_lookup, dataset_summary

def test_patient_lookup():
    row=patient_lookup("P100000")
    assert row is not None
    assert row["patient_id"]=="P100000"

def test_summary():
    s=dataset_summary()
    assert s["rows"]==5000
    assert 0 < s["readmission_rate"] < 1

# tool calling test from agent.tools, checking if patient_lookup and dataset_summary functions return expected results.