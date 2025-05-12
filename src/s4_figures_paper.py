"""Script with helper functions for visualizations"""


## Improt required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textalloc as ta
import networkx as nx
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from scipy.stats import spearmanr

import meneame as mn

def read_data(path, id_twitter, id_pol_watch):
    # read csv
    df = pd.read_csv(path)

    # Add twitter ideology
    df['twitter_ideology'] = df["story"].map(id_twitter)
    df['pol_watch_ideology'] = df["story"].map(id_pol_watch)

    # Normalize all varibles except "story"
    sc = StandardScaler()
    df[df.columns[1:]] = sc.fit_transform(df[df.columns[1:]])
    
    return df

def plot_two_dim(df_1, df_2, top, domains, flip=None, suffix1="_pca_1d", suffix2="_0", columns=["SHEEP", "CA"], d_color=lambda x: "tomato",
                 annotate=True, show_regression="lm", alpha=0.5, s=15, cmap=plt.cm.PuOr):
    """ 
    Scatter plots of two dimensions, annotated and potentially with a regression line

    Args:
        df_1: DataFrame with the first dimension
        df_2: DataFrame with the second dimension
        top: topic to plot
        domains: list of domains to plot
        flip: list of indices to flip the sign of the dimensions (CA and SHEEP do not have a sign per se)
        suffix1: suffix for the first dimension
        suffix2: suffix for the second dimension
        columns: list of columns to plot
        d_color: function to map colors to domains
        annotate: whether to annotate the points
        show_regression: type of regression to show ("lm" or "lowess")
        alpha: transparency of the points
        s: size of the points
        cmap: colormap to use

    Returns:
        d: DataFrame with the two dimensions and the color
    """
    d = pd.concat(
        [
            df_1.set_index("story")[[f"{top}{suffix1}"]],
            df_2.set_index("story")[[f"{top}{suffix2}"]],
        ],
        axis=1,
    )
    d.columns = columns
    d = d.dropna()
    from scipy.stats import spearmanr
    print(spearmanr(d[columns[0]], d[columns[1]]))
    d["color"] = d.index.map(d_color)
    d = d.loc[list(set(domains) & set(d.index))]
    if flip is not None:
        for i in flip:
            d[columns[i]] = -d[columns[i]]
    mn.plot_annotated_scatter(d.index, d[columns[0]], d[columns[1]], c=d["color"],
                        show=False, adjust=annotate, default_color="darkgray", s=s, 
                        show_title=True, marker="o", alpha=alpha, cmap=cmap)
    
    if show_regression == "lm":
        # Regression plot when the color is not nan
        d_reg = d.dropna()
        
        sns.regplot(x=columns[0], y=columns[1], data=d_reg, scatter=False, color="gray", 
                    ci=None, line_kws={"linestyle": "--", "lw": 2, "alpha": 0.8, "zorder": 0 })
    elif show_regression == "lowess":
        d_reg = d.dropna()
        sns.regplot(x=columns[0], y=columns[1], data=d, scatter=False, color="gray", lowess=True, 
                    line_kws={"linestyle": "--", "lw": 2, "alpha": 0.8, "zorder": 0 })
    
    # Print correlation coefficient (spearman)
    corr_s = d.corr(method="spearman").iloc[0, 1]
    corr_p = d.corr(method="pearson").iloc[0, 1]
    print(corr_s, corr_p)
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    # plt.ylim(-0.6, 0.6)
    # plt.xlim(-5, 5)
    #plt.show()

    return d

def compare_external_ideology(df_emb, var="twitter_ideology", var_emb="_pca_1d", top="Politics", flip=None):
    d = df_emb[["story", var, f"{top}{var_emb}"]]
    d = d.dropna()
    if flip:
        d[f"{top}{var_emb}"] = -d[f"{top}{var_emb}"]
    
    mn.plot_annotated_scatter(d["story"], d[f"{top}{var_emb}"], d[var], c=d[var],
                        show=False, adjust=True, default_color="darkgray", s=20, 
                        show_title=True, marker="o", textcolor="k")
    # plt.xticks([])
    plt.xlabel(f"Attitude towards {top}")
    sns.despine(top=False, right=False, left=False, bottom=False)#, color="gray")
    #plt.grid()
    # plt.yticks([])
    # Add regression
    sns.regplot(x=f"{top}{var_emb}", y=var, data=d, scatter=False, color="gray", 
                line_kws={"linestyle": "--", "lw": 2, "alpha": 0.8, "zorder": 0 })
    # Print correlation coefficient (pearson and spearman)
    d = d.set_index("story")
    corr_s = d.corr(method="spearman").iloc[0, 1]
    corr_p = d.corr(method="pearson").iloc[0, 1]
    print(corr_s, corr_p)


def create_subplots_external_ideology(df_emb_sheep_outlets, df_emb_ca_outlets, var_sheep="_pca_1d", var_ca="_0", top="Politics", flip=[False, False], save=True):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)

    compare_external_ideology(df_emb_sheep_outlets, var="twitter_ideology", top=top, flip=flip[0], var_emb=var_sheep)
    plt.ylabel("Ideology (Twitter)", fontsize=8)
    plt.xlabel(f"Attitude towards {top} (SHEEP)", fontsize=8)
    plt.title(f"(A) Using Twitter as ideology source", loc="left", fontsize=8)
    plt.subplot(1, 2, 2)
    compare_external_ideology(df_emb_sheep_outlets, var="pol_watch_ideology", top=top, flip=flip[0], var_emb=var_sheep)
    plt.ylabel("Ideology (Political Watch)", fontsize=8)
    plt.xlabel(f"Attitude towards {top} (SHEEP)", fontsize=8)
    plt.title(f"(B) Using Political Watch as ideology source", loc="left", fontsize=8)
              
    plt.tight_layout()
    if save:
        plt.savefig(f"figures/paper/fig1_external_ideology_outlets_{top}_SHEEP.pdf", bbox_inches="tight")
    plt.show()
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    compare_external_ideology(df_emb_ca_outlets, var="twitter_ideology", top=top, flip=flip[1], var_emb=var_ca)
    plt.ylabel("Ideology (Twitter)", fontsize=8)
    plt.xlabel(f"Attitude towards {top} (CA)", fontsize=8)
    plt.title(f"(A) Using Twitter as ideology source", loc="left", fontsize=8)
    plt.subplot(1, 2, 2)
    compare_external_ideology(df_emb_ca_outlets, var="pol_watch_ideology", top=top, flip=flip[1], var_emb=var_ca)
    plt.ylabel("Ideology (Political Watch)", fontsize=8)
    plt.xlabel(f"Attitude towards {top} (CA)", fontsize=8)
    plt.title(f"(B) Using Political Watch as ideology source", loc="left", fontsize=8)
    plt.tight_layout()
    
    if save: 
        plt.savefig(f"figures/paper/fig1_external_ideology_outlets_{top}_CA.pdf", bbox_inches="tight")
    plt.show()



def plot_network(G, pos, df_emb, var, ax, colorbar=True, norm_color=True, threshold=3):
    """ 
    P
    """
    # # Plot network using as node color the embedding from df_emb_sheep_comments
    d = df_emb.set_index("story")[var].to_dict() 


    # Compute IQR-based color scaling
    russia_pca_values = df_emb[var].dropna()
    mean = np.mean(russia_pca_values)
    std = np.std(russia_pca_values)

    # Set min and max limits for the colormap
    vmin = mean - 2*std
    vmax = mean + 2*std

    print(vmin, vmax, ":", np.min(russia_pca_values), np.max(russia_pca_values))
    nodelist = [node for node in G.nodes if (d.get(node) is not None) and (np.isfinite(d[node]))]  

    color = [d[node] if d.get(node) is not None else 0 for node in nodelist]

    
    # Draw nodes without transparency
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=10, 
        node_color=color, 
        cmap=plt.cm.RdYlBu, 
        vmin=vmin, 
        vmax=vmax,
        nodelist=nodelist,
    
    )
    filtered_edges = [
        (u, v) for u, v in G.edges if u in nodelist and v in nodelist
    ]
    filtered_edges = [(u, v) for u, v in filtered_edges if (np.abs(G[u][v]["weight_full"]) > threshold)]
    if norm_color:
        edge_color = [G[u][v]["color"]  for u, v in filtered_edges]
    else:
        filtered_edges = [(u,v) for u, v in filtered_edges if G[u][v]["color"] == "cornflowerblue"]
        edge_color = [G[u][v]["color"] for u, v in filtered_edges]
    # Draw edges with transparency
    nx.draw_networkx_edges(
        G, 
        pos, 
        edgelist=filtered_edges,
        edge_color=edge_color, 
        alpha=0.4,  # Edge transparency
    )
    if colorbar:
        # Create a ScalarMappable for the colorbar
        sm = mpl.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])  # Required for colorbar

        # Add a colorbar at the top
        plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, label=f"Attitude towards {var.split('_')[0]}")



## Functions for heatmaps
def normalize_by_row_and_column(matrix, tol=1e-6, max_iter=1000):
    """
    Normalize a matrix by rows and columns using iterative proportional fitting.

    Parameters:
    - matrix: 2D numpy array of counts
    - tol: tolerance for convergence
    - max_iter: maximum number of iterations

    Returns:
    - normalized_matrix: 2D numpy array that is normalized by rows and columns
    """
    # Copy the matrix to avoid modifying the original matrix
    norm_matrix = matrix.copy().astype(float)
    for _ in range(max_iter):
        # Row normalization
        row_sums = norm_matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(norm_matrix, row_sums, where=row_sums!=0)
        
        # Column normalization
        col_sums = norm_matrix.sum(axis=0, keepdims=True)
        norm_matrix = np.divide(norm_matrix, col_sums, where=col_sums!=0)
        
        # Check for convergence
        if np.allclose(norm_matrix.sum(axis=1), 1, atol=tol) and np.allclose(norm_matrix.sum(axis=0), 1, atol=tol):
            break
    return norm_matrix

def get_colors(squares, label):
    if label == "SHEEP":
        colors = plt.cm.cividis(np.linspace(0, 1, squares.shape[0]))
    else:
        colors = plt.cm.plasma(np.linspace(0, 1, squares.shape[0]))

    return colors

def plot_vote_matrix(data, topic, df_emb, var="_pca_1d", flip=False, bins=None, type="sum", label="SHEEP", 
                     plot=True, squares=None):
    if squares is not None:
        colors = get_colors(squares, label)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

    ideology = df_emb.set_index("story")[f"{topic}{var}"].dropna()
    if flip:
        ideology = -ideology
    ideology = ideology.to_dict()

    d = data.copy()
    d = d.loc[d["final_topic"]==topic]
    if bins is None:
        bins = np.sqrt(len(d["username_vote"].unique())) // 1.5 # 2
        if bins % 2 != 0:
            bins += 1

    d["u_v_id"] = d["username_vote"].map(ideology)
    d["u_p_id"] = d["username_post"].map(ideology)
    d = d.dropna(subset=["u_v_id", "u_p_id"])
    i = ideology.values()
    # convert to numpy
    i = np.array(list(i))

    bins_perc = np.percentile(i, np.linspace(0, 100, int(bins)))
    #bins_perc = np.linspace(np.min(i), np.max(i), int(bins))

    bins_centers = np.round(1/2*(bins_perc[:-1]+bins_perc[1:]), 2)
    d["u_v_id_cut"] = pd.cut(d["u_v_id"], bins_perc, duplicates='drop', labels=bins_centers, ordered=False)
    d["u_p_id_cut"] = pd.cut(d["u_p_id"], bins_perc, duplicates='drop', labels=bins_centers, ordered=False)

    
    if type == "sum_vote":
        f_agg = "mean"
        d = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["comment_vote"].agg(f_agg).unstack()
        #d = d.groupby(["u_v_id_cut", "u_p_id_cut"])["comment_vote"].agg(f_agg).unstack()
        d = d.T
        if plot:
            sns.heatmap(d,  cmap="RdBu", vmin=-1, vmax=1,
                        cbar_kws={'label': 'Average vote by user attitude'})
            
    if type == "sum_vote_std":
        f_agg = "std"
        d = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["comment_vote"].agg(f_agg).unstack()
        #d = d.groupby(["u_v_id_cut", "u_p_id_cut"])["comment_vote"].agg(f_agg).unstack()
        d = d.T
        if plot:
            sns.heatmap(d,  cmap="afmhot_r", vmin=0, vmax=1.5,
                    cbar_kws={'label': 'STD of vote sign by user attitude'})
    elif type == "sum_user":
        f_agg = "sum"
        d["weight"] = d.groupby("username_vote")["comment_vote"].transform(len)
        d["comment_vote_w"] = d["comment_vote"] * d["weight"]
        
        d_w = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["weight"].agg(f_agg).unstack()
        d = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["comment_vote_w"].agg(f_agg).unstack()

        d = d.div(d_w, axis=0)
        d = d.T

        #d = d.groupby(["u_v_id_cut", "u_p_id_cut"])["comment_vote"].agg(f_agg).unstack()
        if plot:
            sns.heatmap(d,  cmap="RdBu", vmin=-1, vmax=1,
                    cbar_kws={'label': 'Average vote by user attitude'})
    elif type == "len":
        f_agg = len
        d = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["comment_vote"].agg(f_agg).unstack()
        d = d.fillna(0).astype(float)
        d[:] = normalize_by_row_and_column(d.values)
        #d[:] = np.log10(d.values)
        d = d.T
        # Normalize by row
        #d = d.div(d.sum(axis=1), axis=0)
        if plot:
            sns.heatmap(d,  cmap="afmhot_r", vmin=0, vmax=0.5/np.sqrt(len(bins_perc)),
                    cbar_kws={'label': 'Number of votes (normalized) by user attitude'})
    elif type == "len_pos":
        f_agg = len
        d = d.loc[d["comment_vote"]==1]
        d = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["comment_vote"].agg(f_agg).unstack()
        d = d.fillna(0).astype(float)
        d[:] = normalize_by_row_and_column(d.values)
        d = d.T
        # Normalize by row
        #d = d.div(d.sum(axis=1), axis=0)
        if plot:
            sns.heatmap(d,  cmap="afmhot_r", vmin=0, vmax=0.5/np.sqrt(len(bins_perc)),
                    cbar_kws={'label': 'Number of votes (normalized) by user attitude'})
    elif type == "len_neg":
        f_agg = len
        d = d.loc[d["comment_vote"]==-1]
        d = d.groupby(["u_v_id_cut", "u_p_id_cut"], observed=False)["comment_vote"].agg(f_agg).unstack()
        d = d.fillna(0).astype(float)
        d[:] = normalize_by_row_and_column(d.values)
        d = d.T
        # Normalize by row
        #d = d.div(d.sum(axis=1), axis=0)
        if plot:
            sns.heatmap(d,  cmap="afmhot_r", vmin=0, vmax=0.5/np.sqrt(len(bins_perc)),
                    cbar_kws={'label': 'Number of votes (normalized) by user attitude'})
    if plot:
        # ADd all tick labels
        plt.xticks(np.arange(len(bins_centers)), bins_centers)
        plt.yticks(np.arange(len(bins_centers)), bins_centers)

        plt.xlabel(f"Attitude of user (vote) towards {topic} ({label})")
        plt.ylabel(f"Attitude of user (comment) towards {topic} ({label})")
        if squares is not None:
            for i, (min_, max_) in squares.iterrows():
                # Find min when bins_centers is equal to min_
                min_bin = np.where(bins_centers == min_)[0][0]-2
                max_bin = np.where(bins_centers == max_)[0][0]+3
                # plot 
                plt.plot([min_bin, min_bin], [min_bin, max_bin], color=colors[i], lw=3)
                plt.plot([max_bin, max_bin], [min_bin, max_bin], color=colors[i], lw=3)
                plt.plot([min_bin, max_bin], [min_bin, min_bin], color=colors[i], lw=3)
                plt.plot([min_bin, max_bin], [max_bin, max_bin], color=colors[i], lw=3)
                
                
    return d

def find_partitions(A1, A2):
    A_total = pd.concat([A1.T, A2.T], axis=1).fillna(0)
    
    values = A_total.index

    length = np.sqrt((A_total**2).sum(axis=1)).values[:,None]
    A_total = A_total / length

    # Find clusters using KMeans
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    

    sc_max = 0
    for n_cluster in range(2, 6):
        kmeans = KMeans(n_clusters=n_cluster, random_state=2024, n_init="auto").fit(A_total)
        # get silhouette score
        sc = silhouette_score(A_total, kmeans.labels_)
        ch = calinski_harabasz_score(A_total, kmeans.labels_)
        db = davies_bouldin_score(A_total, kmeans.labels_)
        
        #print(n_cluster, sc, ch, db)
        if ch > sc_max:
            sc_max = ch
            labels = kmeans.labels_
            
    
    values_sorted = sorted([(np.mean(values[labels == lab]), lab) for lab in set(labels)])
    d_swap = dict()
    for lab_sorted, (val, lab) in enumerate(values_sorted):
        #print(val, lab, lab_sorted)
        d_swap[lab] = lab_sorted
    labels = [d_swap[lab] for lab in labels]
    

    d_part = dict(zip(values, labels))

    # # Find communities using the Louvain algorithm
    # G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph())
    # partition = nx.community.louvain_communities(G, seed=2024)
    # d_part = {node: i for i, nodes in enumerate(partition) for node in nodes}
    # print(len(partition), "partitions found")
    return d_part

def get_squares(d_part):
    def get_sec_min(x):
        return sorted(list(set(x)))[2]
    def get_sec_max(x):
        return sorted(list(set(x)))[-3]
    # reverse d_partition
    d = (pd.DataFrame.from_dict(d_part, orient="index")
         .reset_index()
         .groupby(0)
         .agg([get_sec_min, get_sec_max]) 
    )
    d.columns = ["min", "max"]
    return d


def plot_hist(id_values, d_partition, square, label):
    colors = get_colors(square, label)
    points_partition = np.array(list(d_partition.keys()))

    id_values[id_values< -2]  = -2
    id_values[id_values> 2]  = 2

    # Plot the distribution of the ideology of the users
    values, bins_plot = np.histogram(id_values, bins=np.arange(np.min(id_values), np.max(id_values), 100/len(id_values)), density=True)
    bar_size = np.diff(bins_plot)[0]

    
    for x, v in zip(bins_plot, values):
        # Find closest point in points partition to v
        closest = np.argmin(np.abs(points_partition - x))
        point = points_partition[closest]
        
        color = colors[d_partition[point]]
        #print(v, closest, point, d_partition[point], color)
        plt.bar(x, v, color=color, alpha=0.5, width=bar_size, edgecolor="none")
    plt.xlabel("Attitude", loc="center")
    plt.ylabel("Share of users")


def plot_range(df, var, data, domains, bin_size=100, adjust_limit=0, correct_votes=True, id=None):
    # Create bins as in the heatmaps
    df = df.loc[~df["story"].isin(domains)].copy().dropna(subset=[var])
    df["cut"] = pd.qcut(df[var], len(df)//bin_size)
    
    # Calculate the average vote and the sum of votes (all users have the same weight) per domain
    domain2vote = (data.loc[(data["final_topic"]==var.split("_")[0])]
                   .groupby(["username_vote", "story_original_url_domain"])["story_vote"].mean() #same weight to all users
                   .groupby("story_original_url_domain").mean()
                   ).to_dict()
    domain2len = (data.loc[(data["final_topic"]==var.split("_")[0])]
                    .groupby(["username_vote", "story_original_url_domain"])["story_vote"].mean() #same weight to all users
                    .groupby("story_original_url_domain").count()
                    )
    
    # Normalize length (for normalization later on)
    domain2len /= domain2len.sum()
    domain2len = domain2len.to_dict()
                
    
    values = []
    # For each bin
    for i, d in df.groupby("cut"):
        users = set(d["story"])

        # CAlculate theaverage vote per domain in the ideological bin
        u = (data.loc[(data["final_topic"]==var.split("_")[0]) & (data["username_vote"].isin(set(users)))]
            .groupby(["username_vote", "story_original_url_domain"])[["story_vote"]]
            .agg([np.mean, len])
            .reset_index()
        )

        u.columns = ["username_vote", "story_original_url_domain", "story_vote", "len"]
        
        # Compare to average for all users 
        u["av_story_vote"] = u["story_original_url_domain"].map(domain2vote)
        u["story_vote"] = u["story_vote"] - u["av_story_vote"]

        # Scale by the relative frequency of the domain
        u["av_len"] = u["story_original_url_domain"].map(domain2len)
        #u["len_total"] = u["len"].copy()/u["len"].sum()
        u["diff_len"] = (u["len"]/u["av_len"])
        
        #u["story_vote2"] = u["story_vote"].copy()

        if correct_votes:
            #correct by changes in propensity to vote
            u["story_vote"] = u["story_vote"]*u["diff_len"]
            #diff_len>0 (more votes) --> give extra weight to the story_vote
            #diff_len<0 (less votes) --> give less weight to the story_vote

        #u2 = u.groupby("story_original_url_domain")["story_vote2"].agg(["mean", "std", len])

        u = u.groupby("story_original_url_domain")["story_vote"].agg(["mean", "std", len])
        if id is None:
            u["tw"] = u.index.map(mn.domain_ideology_twitter)
        else:
            u["tw"] = u.index.map(id)
        #print(i, len(d), d[var].min(), d[var].max())
        #display(u.dropna())
        values.append((u["tw"]*u["mean"]*u["len"]).sum()/u["len"].sum())
        #values2.append((u2["mean"]*u2["len"]).sum()/u2["len"].sum())
        #break
    
    # extract middle bins from qcut
    cuts = df["cut"].dropna().unique()[::-1]
    # exctract the middle values
    cuts = sorted([c.mid for c in cuts])
    
    #values.append(values[-1])
    #cuts = sorted(set([c.left for c in cuts]+[c.right for c in cuts]))
    
    #sns.regplot(x=cuts, y=values, order=2, ci=None)
    #values2 = np.array(values2)*np.max(values)
    plt.scatter(cuts, values, s=20)
    #plt.scatter(cuts, values2, s=20, color="tomato")
    # Regression lowess line
    sns.regplot(x=cuts, y=values, scatter=False, color="gray", lowess=True, 
                line_kws={"linestyle": "--", "lw": 2, "alpha": 0.8, "zorder": 0 })

    # Add Spearman correlation to the top left area of the plot
    corr_s = spearmanr(cuts, values)[0]
    
    plt.text(0.05, 0.95, f"Spearman corr: {corr_s:.1%}", ha="left", va="top", 
             transform=plt.gca().transAxes, fontsize=10)

    #u.sort_values(by="len", ascending=False).head(20).sort_values(by="mean", ascending=False)

    # adjust limit
    xlim = 1.05*np.quantile(cuts, [adjust_limit, 1-adjust_limit])
    #plt.xlim(*xlim)
    plt.plot(xlim, [0, 0], color="lightgray", linestyle="--", zorder=0)
    plt.xlim(*xlim)
    
    plt.ylabel("Attitude (based on stories voted)")


        