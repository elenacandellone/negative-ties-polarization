from datetime import datetime
import pandas as pd
import json
import yaml
from tldextract import extract


def load_yaml_headers(path, filename):
    with open(f"{path}/file_headers.yml", "r") as f:
        headers = yaml.safe_load(f)
    if filename not in headers:
        raise ValueError(f"Headers for '{filename}' not found in YAML.")
    return headers[filename]


def read_tsv_file(path, filename, headers, user_map):
    df = pd.read_csv(f"{path}/{filename}", sep="\t", header=None, on_bad_lines="warn", dtype=str)
    df.columns = headers
    df = df.drop_duplicates().dropna(subset=["username"])
    df["username"] = df["username"].map(user_map)
    return df


def parse_dates(df, column, fmt):
    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce')
    return df


def map_story_metadata(df_stories_urls, df_stories_votes):
    df_stories_urls = df_stories_urls.rename(columns={"username": "username_post"})
    df_stories_urls["story_original_url_domain"] = df_stories_urls["story_original_url"].apply(lambda x: extract(x).domain)

    story_to_user = df_stories_urls.set_index("story_id")["username_post"].to_dict()
    df_stories_votes = df_stories_votes.rename(columns={"username": "username_vote"})
    df_stories_votes["username_post"] = df_stories_votes["story_id"].map(story_to_user)
    
    return df_stories_urls, df_stories_votes


def map_comment_metadata(df_comments_urls, df_comments_votes, story_url_to_id):
    df_comments_urls = df_comments_urls.rename(columns={"username": "username_post"})
    df_comments_urls["story_id"] = df_comments_urls["story_url"].map(story_url_to_id)

    comment_to_user = df_comments_urls.set_index("comment_id")["username_post"].to_dict()
    comment_to_story = df_comments_urls.set_index("comment_id")["story_id"].to_dict()

    df_comments_votes["comment_id"] = df_comments_votes["comment_id"].str.split("=").str[-1]
    df_comments_votes = df_comments_votes.rename(columns={"username": "username_vote"})
    df_comments_votes["username_post"] = df_comments_votes["comment_id"].map(comment_to_user)
    df_comments_votes["story_id"] = df_comments_votes["comment_id"].map(comment_to_story)

    return df_comments_urls, df_comments_votes


def main():
    path_data = "./data_raw"
    path_out = "./data_snapshot"
    path_keys = "./data_keys"
    date_min = pd.to_datetime("2022-11-27")

    with open(f"{path_data}/username_to_id.json", "r") as f:
        user_to_id = json.load(f)

    # Read all files
    file_names = ["comments_urls.tsv", "comments_votes.tsv", "stories_urls.tsv", "stories_votes.tsv"]
    dfs = {name: read_tsv_file(path_data, name, load_yaml_headers(path_data, name), user_to_id)
           for name in file_names}

    # Rename for convenience
    df_comments_urls = dfs["comments_urls.tsv"]
    df_comments_votes = dfs["comments_votes.tsv"]
    df_stories_urls = dfs["stories_urls.tsv"]
    df_stories_votes = dfs["stories_votes.tsv"]

    # Enrich and clean data
    df_stories_urls, df_stories_votes = map_story_metadata(df_stories_urls, df_stories_votes)
    story_url_to_id = df_stories_urls.set_index("story_url")["story_id"].to_dict()
    df_comments_urls, df_comments_votes = map_comment_metadata(df_comments_urls, df_comments_votes, story_url_to_id)

    # Parse all date fields
    df_comments_votes = parse_dates(df_comments_votes, "comment_vote_time", "%d-%m-%Y %H:%M:%S")
    df_comments_urls = parse_dates(df_comments_urls, "comment_time", "%Y-%m-%d %H:%M:%S")
    df_stories_votes = parse_dates(df_stories_votes, "story_vote_time", "%d-%m-%Y %H:%M")
    df_stories_urls = parse_dates(df_stories_urls, "story_submitted_time", "%Y-%m-%dT%H:%M:%S")
    df_stories_urls = parse_dates(df_stories_urls, "story_published_time", "%Y-%m-%dT%H:%M:%S")

    # Filter by minimum date
    df_stories_urls = df_stories_urls[df_stories_urls["story_submitted_time"] > date_min]

    # Filter all other datasets to only keep relevant stories
    valid_story_ids = set(df_stories_urls["story_id"])
    df_comments_votes = df_comments_votes[df_comments_votes["story_id"].isin(valid_story_ids)]
    df_comments_urls = df_comments_urls[df_comments_urls["story_id"].isin(valid_story_ids)]
    df_stories_votes = df_stories_votes[df_stories_votes["story_id"].isin(valid_story_ids)]

    print(len(df_comments_votes), len(df_stories_votes), len(df_comments_urls), len(df_stories_urls))
    # Save files
    df_comments_urls.to_csv(f"{path_out}/df_comments_urls.tsv.gz", sep="\t", index=None, compression="gzip")
    df_comments_votes.to_csv(f"{path_out}/df_comments_votes.tsv.gz", sep="\t", index=None, compression="gzip")
    df_stories_urls.to_csv(f"{path_out}/df_stories_urls.tsv.gz", sep="\t", index=None, compression="gzip")
    df_stories_votes.to_csv(f"{path_out}/df_stories_votes.tsv.gz", sep="\t", index=None, compression="gzip")

    # Copy keys to safe folder
    import shutil
    shutil.copyfile(f"{path_data}/id_to_username.json", f"{path_keys}/id_to_username.json")   


if __name__ == "__main__":
    main()
    
