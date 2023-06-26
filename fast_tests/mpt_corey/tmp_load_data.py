import boto3

bucket_name = "det-llm-hackathon"
local_folder = "/cstor/coreystaten/data/hackathon"


def download_files_from_s3_bucket(bucket_name, local_folder):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)
    for page in pages:
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_file_path = os.path.join(local_folder, key)
            if os.path.exists(local_file_path):
                print(f"Skipping {key} because {local_file_path} already exists.")
                continue
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            print(f"Downloading {key} to {local_file_path}")
            s3.download_file(bucket_name, key, local_file_path)
