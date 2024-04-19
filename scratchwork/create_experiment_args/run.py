import pathlib

from determined.experimental import client

if __name__ == "__main__":
    config = {
        "resources": {"slots_per_trial": 0},
        "entrypoint": "echo 'Hello'",
        "searcher": {"name": "single", "max_length": 1, "metric": "arbitrary"},
    }

    parent_dir = pathlib.Path(__file__).parent.resolve()
    print(f"{parent_dir=}")
    client.create_experiment(config, parent_dir, template="template")
