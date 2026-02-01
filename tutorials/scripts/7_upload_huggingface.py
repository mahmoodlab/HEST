from huggingface_hub import HfApi


if __name__ == '__main__':
    version_name = 'v1.3.0'
    input_path = "/media/paul/ssd2/HEST_results"

    api = HfApi()

    # res = api.create_pull_request(
    #     repo_id="MahmoodLab/hest",
    #     repo_type="dataset",
    #     title=version_name
    # )
    

    api.upload_large_folder(
        folder_path=input_path,
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        revision='refs/pr/26'
    )

    # api.upload_file(
    #     path_or_fileobj="/home/paul/Downloads/README.md",
    #     path_in_repo="README.md",
    #     repo_id="MahmoodLab/hest",
    #     repo_type="dataset",
    #     revision='refs/pr/23'
    # )