
from feast import FeatureStore

store = FeatureStore(repo_path="pipeline/sepsis_feast/feature_repo")

result = store.get_online_features(
    features=[
        "patient_vitals:heart_rate",
        "patient_vitals:shock_index",
        "patient_vitals:hr_rolling_mean",
    ],
    entity_rows=[{"patient_id": "p000001"}],
).to_dict()

print(result)