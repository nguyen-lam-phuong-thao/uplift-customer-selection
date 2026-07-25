import dagshub
import mlflow


def setup_mlflow():
    dagshub.init(
        repo_owner="nguyen-lam-phuong-thao",
        repo_name="uplift-customer-selection",
        mlflow=True,
    )

    mlflow.set_experiment("criteo-uplift-modeling")