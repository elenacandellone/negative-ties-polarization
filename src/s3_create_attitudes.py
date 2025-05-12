"""Functions used in the notebook 3_create_attitudes.ipynb to create the network embeddings"""

# Required imports in the script
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import meneame as mn

def print_stats(data, type_vote="story"):
    """1_topic_modelling.ipynb

    Print statistics of the data by topic. e.g. Crime: N. users: 4903, N. outlets/commenters: 315. Votes: 45004 (+1), 2818 (-1)

    Args:
        data (pd.DataFrame): Dataframe containing the data.
        type_vote (str): Type of vote. Can be either "story" or "comment".

    Returns:
        None
    """
    if type_vote == "story":
        groupby = "story_original_url_domain"
    else:
        groupby = "username_post"

    for topic, d in data.groupby("final_topic"):
        # filter users by activity
        d["min_votes_from_user"] = d.groupby("username_vote")[f"{type_vote}_vote"].transform(len)
        d["min_votes_to_users_or_domain"] = d.groupby(groupby)[
            f"{type_vote}_vote"
        ].transform(len)
        d["min_comments_or_stories"] = d.groupby(groupby)[
            f"{type_vote}_id"
        ].transform(lambda x: len(set(x)))

        filtered_df = (
            d.groupby(["username_vote", groupby])
            .agg({f"{type_vote}_vote": "sum", f"{type_vote}_vote_time": "min"})
            .reset_index()
        )
        filtered_df = filtered_df.rename(
            columns={f"{type_vote}_vote": "weight", f"{type_vote}_vote_time": "event_time"}
        )
        filtered_df

        users = len(filtered_df["username_vote"].unique())
        outlets = len(filtered_df[groupby].unique())
        votes = (filtered_df["weight"] < 0).value_counts().values

        print(
            f"{topic}: N. users: {users}, N. outlets/commenters: {outlets}. Votes: {votes[0]} (+1), {votes[1]} (-1)"
        )

def calculate_embeddings(data, path_save_embeddings, min_comments_or_stories=10, norm_laplacian=False, bipartite=True, last_suffix=""):
    """
    Calculate the embeddings for the data.
    Args:
        data (pd.DataFrame): Dataframe containing the data.
        min_comments_or_stories (int): Minimum number of comments or stories.
        norm_laplacian (bool): Whether to normalize the Laplacian.
        bipartite (bool): Whether to use bipartite graph.
        last_suffix (str): Suffix to add to the filename.
    Returns:
        None
    """
    if bipartite:
        suffix = "bipartite"
    else:
        suffix = "unipartite"

    # Create SHEEP embeddings
    df_all_embeddings_sheep = mn.create_embeddings(
        data,
        method="sheep",
        bipartite=bipartite,
        min_votes_from_user=10,
        min_votes_to_users_or_domains=0,
        min_comments_or_stories=min_comments_or_stories,
        plot_sheep=True,
        normalize_laplacian=norm_laplacian, 
        adjust_weight=False,
    )

    # Create CA embeddings
    df_all_embeddings_ca = mn.create_embeddings(
        data,
        method="ca",
        bipartite=bipartite,
        min_votes_from_user=10,
        min_votes_to_users_or_domains=0,
        min_comments_or_stories=min_comments_or_stories,
    )
    
    # Concatenate embeddings and save to disk
    df_all_embeddings_sheep = df_all_embeddings_sheep.reset_index().rename(
        columns={"user": "story"}
    )
    df_all_embeddings_sheep.to_csv(f"{path_save_embeddings}/embeddings_sheep_{suffix}_normed_{norm_laplacian}_min_votes_{min_comments_or_stories}{last_suffix}.csv", index=False)

    # For consistency in future functions the columns as "story", even if they are users
    df_all_embeddings_ca = df_all_embeddings_ca.reset_index().rename(
        columns={"user": "story"}
    )
    df_all_embeddings_ca.to_csv(f"{path_save_embeddings}/embeddings_ca_{suffix}_min_votes_{min_comments_or_stories}{last_suffix}.csv", index=False)

    # Remove outliers (over 50 times the IQR) and impute missing values
    df_all_embeddings_sheep_m, outliers_sheep = add_values(df_all_embeddings_sheep)
    df_all_embeddings_ca_m, outliers_ca = add_values(df_all_embeddings_ca)
    df_all_embeddings_sheep_m.to_csv(f"{path_save_embeddings}/embeddings_sheep_{suffix}_normed_{norm_laplacian}_no_outliers_min_votes_{min_comments_or_stories}{last_suffix}.csv", index=False)
    df_all_embeddings_ca_m.to_csv(f"{path_save_embeddings}/embeddings_ca_{suffix}_no_outliers_min_votes_{min_comments_or_stories}{last_suffix}.csv", index=False)


def run_pca(path, topics, domains):
    """
    Run PCA on the embeddings and save the results to disk.
    Args:
        path (str): Path to the embeddings file.
        topics (list): List of topics.
        domains (list): List of domains.
    Returns:

    """
    df_emb = pd.read_csv(path)
    if "_sheep_" in path:
        path_save = path.replace("sheep", "pca_sheep")
        emb = "sheep"
    else:
        path_save = path.replace("ca", "pca_ca")
        emb = "ca"

    df_pca_emb = mn.create_pca_emb(
        df_emb,
        topics=topics, domains=domains, normalize=True, emb=emb
    )

    df_pca_emb.to_csv(path_save, index=False)


def remove_outliers(a):
    """
    Remove outliers from the data using the IQR method.

    Args:
        a (pd.Series): Data to remove outliers from.
    Returns:
        pd.Series: Data with outliers removed.
    """
    q75, q25 = np.percentile(a[np.isfinite(a)], [75, 25])
    iqr = q75 - q25
    ub = a.median() + 50 * iqr
    lb = a.median() - 50 * iqr
    # print(np.sum((a>ub) | (a<lb)), np.sum((a>ub) | (a<lb))/len(a))
    a[(a > ub) | (a < lb)] = np.nan
    return a


def add_values(df):
    """
    Remove outliers from the data using the IQR method and impute missing values using KNN (the embeddings are highly correlated).
    Args:
        df (pd.DataFrame): Data to remove outliers from.
    Returns:
        pd.DataFrame: Data with outliers removed and missing values imputed.
    """

    from collections import Counter

    df = df.set_index("story")

    list_outliers = []
    knni = KNNImputer(n_neighbors=10)
    nan_values = np.isnan(df.values)

    df = df.apply(remove_outliers)
    # print outliers as those with misisng vlaues
    for i, col in enumerate(df.columns):
        outliers = list(df.loc[np.isnan(df[col]) & ~nan_values[:, i]].index)
        list_outliers += outliers

    df[:] = knni.fit_transform(df)
    df[nan_values] = np.nan
    # print(Counter(list_outliers).most_common())
    
    return df.reset_index(), set(list_outliers)
