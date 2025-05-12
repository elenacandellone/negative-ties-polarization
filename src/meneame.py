# Function to strip accents from text
import itertools
import unicodedata
import string
import pandas as pd
import networkx as nx
import numpy as np

# viz
import pylab as plt
import seaborn as sns
import textalloc as ta
from scipy.stats import spearmanr, pearsonr
import matplotlib.colors

#do a PCA onto one dimension, print correlation 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from multiprocess import Pool

def opposing_lap(G):

    selected_edges = [(u,v) for u,v in G.edges if G[u][v]['weight'] < 0]

    G_neg = nx.Graph()
    G_neg.add_nodes_from(G)
    G_neg.add_edges_from(selected_edges)
    degree_neg = [val for (node, val) in G_neg.degree()]
    D_neg = np.diag(degree_neg)

    selected_edges = [(u,v) for u,v in G.edges if G[u][v]['weight'] > 0]

    G_pos = nx.Graph()
    G_pos.add_nodes_from(G)
    G_pos.add_edges_from(selected_edges)
    degree_pos = [val for (node, val) in G_pos.degree()]
    D_pos = np.diag(degree_pos)
    
    A = nx.to_numpy_array(G)
    
    return D_pos + D_neg - A

def repelling_lap(G):

    selected_edges = [(u,v) for u,v in G.edges if G[u][v]['weight'] == -1]

    G_neg = nx.Graph()
    G_neg.add_nodes_from(G)
    G_neg.add_edges_from(selected_edges)
    degree_neg = [val for (node, val) in G_neg.degree()]
    D_neg = np.diag(degree_neg)

    selected_edges = [(u,v) for u,v in G.edges if G[u][v]['weight'] == 1]

    G_pos = nx.Graph()
    G_pos.add_nodes_from(G)
    G_pos.add_edges_from(selected_edges)
    degree_pos = [val for (node, val) in G_pos.degree()]
    D_pos = np.diag(degree_pos)
    
    A = nx.to_numpy_array(G)
    
    return D_pos - D_neg - A


def null_graph_signed(G): 
    d = nx.get_edge_attributes(G, "weight")
    
    shuffled = list(d.values())
    random.shuffle(shuffled)
    shuf_signs = dict(zip(d, shuffled))
    
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    
    nx.set_edge_attributes(H, shuf_signs, "weight")   
    
    return H


def dist(P,k):
    a = 0
    for i in range(k):
        a = a + (P[:,i]-P[:,i][:,np.newaxis])**2
    return np.sqrt(a)

def energy_repelling_lap_norm(P,A,k):
    
    P = P.reshape((-1, k))
    D_un = dist(P,k)**2  
    norm = np.sqrt(np.sum(D_un**2))
    D = D_un/norm  
    
    filter_arr_pos = A > 0
    D_pos = D[filter_arr_pos]
    spring_energy = (D_pos).sum()
    
    filter_arr_neg = A < 0
    D_neg = D[filter_arr_neg]
    anti_spring_energy = (-1)*(D_neg).sum()
    
    return (spring_energy + anti_spring_energy)



def energy_repelling_lap_norm_bis(P, A, k):
    
    P = P.reshape((-1, k))
    D_un = dist(P,k)**2  
    norm = np.sqrt(np.sum(D_un**2))
    D = D_un/norm  
    
    energy = np.multiply(D, A).sum()

    return energy


#check what happens when we remove the weights 

def masked_weights_array(G):
    n = nx.number_of_nodes(G)
    A = nx.to_numpy_array(G)
    new_mat = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if A[i][j] > 0:
                new_mat[i][j] = 1
            elif A[i][j] < 0: 
                new_mat[i][j] = -1
    return new_mat


def sponge(G):

    selected_edges = [(u,v) for u,v in G.edges if G[u][v]['weight'] < 0]

    G_neg = nx.Graph()
    G_neg.add_nodes_from(G)
    G_neg.add_edges_from(selected_edges)
    degree_neg = [val for (node, val) in G_neg.degree()]
    D_neg = np.diag(degree_neg)
    A_neg = nx.to_numpy_array(G_neg)
    L_neg = D_neg - A_neg

    
    selected_edges = [(u,v) for u,v in G.edges if G[u][v]['weight'] > 0]
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G)
    G_pos.add_edges_from(selected_edges)
    degree_pos = [val for (node, val) in G_pos.degree()]
    D_pos = np.diag(degree_pos)
    A_pos = nx.to_numpy_array(G_pos)
    L_pos = D_pos - A_pos
    
    
    N = L_neg + D_pos
    D, V = scipy.linalg.eigh(N)
    Bs = (V * np.sqrt(D)) @ V.T
    M1 = np.linalg.inv(Bs)
    M2 = L_pos + D_neg
    
    step1 = np.matmul(M1, M2)
    step2 = np.matmul(step1, M1)
    
    return step2

# change the default options of visualization
text_color = "#404040"
custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False, "axes.spines.bottom": False,
                "lines.linewidth": 2, "grid.color": "lightgray", "legend.frameon": False,
                 "xtick.labelcolor": text_color, "ytick.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color,"text.color": text_color,
                "axes.labelcolor": text_color, "axes.titlecolor":text_color,"figure.figsize": [5,3],
                "axes.titlelocation":"left","xaxis.labellocation":"left","yaxis.labellocation":"bottom"}

palette = ["#3d348b","#e6af2e","#191716","#e0e2db"] #use your favourite colours
sns.set_theme(context='paper', style='white', palette=palette, font='Verdana', font_scale=1.1, color_codes=True,
rc=custom_params)
           


def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
        .encode('ascii', 'ignore')\
        .decode("utf-8")

    return text

# Function for text preprocessing
def text_preprocessing(df_series):
    # Convert to lowercase and remove digits
    df_series = df_series.str.lower().str.replace(r'\d+', '', regex=True)

    # Define a translator to remove punctuation
    translator = str.maketrans(' ', ' ',string.punctuation+'¿¡')
    # Remove punctuation from text
    df_series = df_series.apply(lambda x: x.translate(translator))
    df_series = df_series.apply(lambda x: strip_accents(x))
    
    return df_series

# Created by hand based on the dendogram of topics
topics_map = {0: '0_agg', 1: '1_agg', 2: '2_agg', 21: '2_agg', 8: '8_agg', 16: '8_agg', 9: '9_agg', 77: '9_agg', 364: '9_agg', 11: '11_agg', 27: '11_agg', 12: '12_agg', 20: '12_agg', 57: '12_agg', 113: '12_agg', 14: '14_agg', 43: '14_agg', 85: '14_agg', 106: '14_agg', 179: '14_agg', 192: '14_agg', 210: '14_agg', 244: '14_agg', 15: '15_agg', 17: '17_agg', 18: '18_agg', 45: '18_agg', 62: '18_agg', 76: '18_agg', 128: '18_agg', 141: '18_agg', 19: '19_agg', 46: '19_agg', 56: '19_agg', 23: '23_agg', 33: '23_agg', 82: '23_agg', 101: '23_agg', 50: '50_agg', 131: '50_agg', 182: '50_agg', 215: '50_agg', 271: '50_agg', 277: '50_agg', 287: '50_agg', 304: '50_agg', 311: '50_agg', 370: '50_agg', 379: '50_agg', 148: '11_agg', 274: '11_agg'}

topic_names = {'0_agg': 'Rusia',
 '1_agg': 'Health_strike',
 '2_agg': 'Climate_change',
 '8_agg': 'Crime',
 '9_agg': 'Sex_assult',
 '11_agg': 'Renewable',
 '12_agg': 'Inflation',
 '14_agg': 'Elections',
 '15_agg': 'Israel',
 '17_agg': 'COVID',
 '18_agg': 'Economy_worker_rights',
 '19_agg': 'Musk_crypto',
 '23_agg': 'Women_rights',
 '50_agg': 'Politics_left'}

domain_ideology_twitter = {'rtve': 0.3868964250776702,
 'abc': 1.0556255774547407,
 'elmundo': 0.921682215146072,
 'atresplayer': 0.2677484675039739,
 'cope': 0.952508323800797,
 'okdiario': 1.3135274405055524,
 'larazon': 1.0232167619216739,
 'ondacero': 0.4437883261745283,
 'telecinco': 0.6326196165819746,
 'vozpopuli': 1.0893794744754357,
 #'youtube': -0.29474299606340904,
 'elespanol': 0.7850623676637541,
 'europapress': -0.08430734821271864,
 'elconfidencial': 0.4219609060987241,
 'telemadrid': 0.214684947340613,
 'cuatro': 0.34341699332782794,
 'canalsur': 0.6305394863891609,
 'eltorotv': 1.3440361158465317,
 'elindependiente': 0.8470249162062686,
 'cadenaser': -0.5225862946092765,
 'eleconomista': 0.6546824709742022,
 'elpais': -0.462315963682772,
 'lavanguardia': -0.13756625801646055,
 'esdiario': 0.7366953433673963,
 'libertaddigital': 1.2673283041540075,
 '20minutos': -0.37805605248100693,
 'elperiodico': -0.029055025171349398,
 'lasexta': -0.39451996894104424,
 'lavozdegalicia': 0.5203767340940603,
 'eldiario': -0.7767191737879745,
 'huffingtonpost': -0.651824369078785,
 #'facebook': -0.4827037446085363,
 #'twitch': -0.713214135324533,
 'infolibre': -0.8431029980942684,
 'publico': -0.8528178051301829,
 'laultimahora': -0.9825709225880228,
 'elsaltodiario': -0.8699422628132775,
 'gaceta': 1.338533449206644}


def read_stories_topics(path_data, topic_file = "2023_10_11_stories_topics.tsv"):
    """
    Reads and combines comment votes and topics data from specified file paths.

    Parameters:
    - path_data (str): The file path where the data files are located.
    - topic_file (str, optional): The file name of the topics
    
    The function performs the following operations:
    - Reads the comment votes data from a gzip compressed TSV file, parsing dates in the 'comment_vote_time' column.
    - Reads the topics data from a TSV file.
    - Merges the comments votes DataFrame with the topics DataFrame based on common keys, defaults to a left join.
    - Sorts the resulting DataFrame by 'comment_vote_time'.
    - Checks and displays any stories that did not merge correctly.
    - Prints the count of positive and negative votes.
    - Prints the time range of the data.

    Returns:
    - A Pandas DataFrame with the merged comment votes and topics, sorted by 'comment_vote_time'.
    """
    
    
    # Read comment votes and stories urls
    df_stories_votes = pd.read_csv(f"{path_data}/df_stories_votes.tsv.gz", sep="\t", compression="gzip", parse_dates=["story_vote_time"])
    
    df_stories_urls = pd.read_csv(f"{path_data}/df_stories_urls.tsv.gz", sep="\t", compression="gzip")
    
    # Read topics
    df_topics = pd.read_csv(f"{path_data}/{topic_file}", sep="\t")
    
    # Read topics
    df_topics = pd.read_csv(f"{path_data}/{topic_file}", sep="\t")
    
    # Add topics to comment  votes
    df_stories_votes = pd.merge(df_stories_votes, df_topics, indicator=True, how="left")
    df_stories_votes = df_stories_votes.sort_values(by="story_vote_time")
    
    stories_to_domain = dict(zip(df_stories_urls.story_id,df_stories_urls.story_original_url_domain))
    df_stories_votes['story_original_url_domain'] = df_stories_votes['story_id'].apply(lambda x: stories_to_domain.get(x))
    
    # Make sure that all stories have been merged
    display(df_stories_votes.loc[df_stories_votes["_merge"] != "both"])
    
    print("Number of positive and negative votes\n", df_stories_votes.story_vote.value_counts())
    print(f"Data time range {df_stories_votes['story_vote_time'].min()}, {df_stories_votes['story_vote_time'].max()}")

    return df_stories_votes


def read_comments_topics(path_data, topic_file = "2024_03_05_stories_final_topics.tsv", bipartite=False):
    """
    Reads and combines comment or story votes and topics data from specified file paths.

    Parameters:
    - path_data (str): The file path where the data files are located.
    - topic_file (str, optional): The file name of the topics
    
    The function performs the following operations:
    - Reads the comment votes data from a gzip compressed TSV file, parsing dates in the 'comment_vote_time' column.
    - Reads the topics data from a TSV file.
    - Merges the comments votes DataFrame with the topics DataFrame based on common keys, defaults to a left join.
    - Sorts the resulting DataFrame by 'comment_vote_time'.
    - Checks and displays any stories that did not merge correctly.
    - Prints the count of positive and negative votes.
    - Prints the time range of the data.

    Returns:
    - A Pandas DataFrame with the merged comment votes and topics, sorted by 'comment_vote_time'.
    """
    
    
    # Read comment votes and stories urls
    if bipartite:
        vote_time = "story_vote_time"
        path = "df_stories_votes.tsv.gz"
    else:
        vote_time = "comment_vote_time"
        path = "df_comments_votes.tsv.gz"

    df_comments_votes = pd.read_csv(f"{path_data}/{path}", sep="\t", compression="gzip", parse_dates=[vote_time])
    df_stories_urls = pd.read_csv(f"{path_data}/df_stories_urls.tsv.gz", sep="\t", compression="gzip")
    
    # Read topics
    df_topics = pd.read_csv(f"{path_data}/{topic_file}", sep="\t")
    
    # Add topics to comment  votes
    df_comments_votes = pd.merge(df_comments_votes, df_topics, indicator=True, how="left")
    df_comments_votes = df_comments_votes.sort_values(by=vote_time)

    stories_to_domain = dict(zip(df_stories_urls.story_id,df_stories_urls.story_original_url_domain))
    stories_to_published = dict(zip(df_stories_urls.story_id,df_stories_urls["story_published_time"].fillna("-9") != "-9"))

    df_comments_votes['story_original_url_domain'] = df_comments_votes['story_id'].apply(lambda x: stories_to_domain.get(x))
    df_comments_votes['published_story'] = df_comments_votes['story_id'].apply(lambda x: stories_to_published.get(x))
    
    # Make sure that all stories have been merged
    #display(df_comments_votes.loc[df_comments_votes["_merge"] != "both"])
    
    #print("Number of positive and negative votes\n", df_comments_votes[vote_time[:-5]].value_counts())
    print(f"Data time range {df_comments_votes[vote_time].min()}, {df_comments_votes[vote_time].max()}")

    return df_comments_votes

# Create graph from comment data (use read_comments_topics to read the data)
def meneame_graph_aggregate(data, topic_no, max_time=None, min_time=None, only_positive_votes=False, filter_largest_component=True, min_votes_from_user=10, min_votes_to_users_or_domains=100, bipartite=False, min_comments_or_stories=10, adjust_weight=False): 
    """
    Aggregates comment data into a graph based on a specified topic number and other filters.

    Parameters:
    - data (DataFrame): The input data containing comment interactions. --> Create it using read_comments_topics()
    - topic_no (int): The topic number to filter the graph data.
    - max_time (datetime, optional): The maximum time for comments to be included in the graph. Defaults to the latest time in data.
    - only_positive_votes (bool, optional): If set to True, only includes positive votes in the graph. Defaults to False.
    - filter_largest_component (bool, optional): If set to True, filters the graph to only include the largest connected component. Defaults to False.
    - min_votes_from_user (int, optional): Minimum number of votes casted by a user. Defaults to 10.
    - min_votes_to_users_or_stories (int, optional): Minimum number of votes casted to a user or domain (if bipartite). Defaults to 100.

    The function performs the following operations:
    - Filters the input data by 'comment_vote_time' and 'topic'.
    - If 'only_positive_votes' is True, further filters the data to include only positive votes.
    - It filters the data, keeping only active users (defined by min_votes_from_user and min_votes_to_users_or_stories)
    - Aggregates the data at the pair level (username_vote to username_post) to calculate the sum of 'comment_vote' and the minimum 'comment_vote_time'.
    - Renames the aggregated columns to 'weight' and 'event_time'.
    - Creates a NetworkX graph from the aggregated DataFrame.
    - If 'filter_largest_component' is True, modifies the graph to keep only the largest connected component.

    Returns:
    - A NetworkX Graph object constructed from the aggregated data.
    """
    if bipartite:
        agg_by = ["username_vote", "story_original_url_domain"]
        type = "story"
    else:
        agg_by = ["username_vote", "username_post"]
        type = "comment"
        
    if max_time is None:
        max_time = data[f"{type}_vote_time"].max()
    if min_time is None:
        min_time = data[f"{type}_vote_time"].min()
    
    # filter by time and topic
    graph_data = data.loc[(data[f"{type}_vote_time"] <= max_time) & (data[f"{type}_vote_time"] >= min_time) & (data["topic"] == topic_no)].copy()

    # filter users by activity
    graph_data["min_votes_from_user"] = graph_data.groupby(agg_by[0])[f"{type}_vote"].transform(len)
    graph_data["min_votes_to_users_or_domain"] = graph_data.groupby(agg_by[1])[f"{type}_vote"].transform(len)
    graph_data["min_comments_or_stories"] = graph_data.groupby(agg_by[1])[f"{type}_id"].transform(lambda x: len(set(x)))
    
    graph_data = graph_data.loc[graph_data["min_votes_from_user"]>min_votes_from_user]
    graph_data = graph_data.loc[graph_data["min_votes_to_users_or_domain"]>min_votes_to_users_or_domains]
    graph_data = graph_data.loc[graph_data["min_comments_or_stories"]>min_comments_or_stories]

    if adjust_weight:
        pos_votes = (graph_data[f"{type}_vote"] == 1).sum()
        neg_votes = (graph_data[f"{type}_vote"] == -1).sum()
        graph_data.loc[graph_data[f"{type}_vote"] == -1, f"{type}_vote"] *= pos_votes/neg_votes
    # keep only positive votes
    if only_positive_votes:
        graph_data = graph_data.loc[graph_data[f"{type}_vote"] == 1]
        
    
    #aggregate data at pair level
    filtered_df = graph_data.groupby(agg_by).agg({f"{type}_vote": "sum", f"{type}_vote_time": "min"}).reset_index()
    filtered_df = filtered_df.rename(columns = {f"{type}_vote": "weight", f"{type}_vote_time": "event_time"})

    # create network
    G = nx.from_pandas_edgelist(
        filtered_df,
        source=agg_by[0],
        target=agg_by[1],
        edge_attr=True,  # This will include all remaining columns as edge attributes
        create_using=nx.Graph()
    )
    

    # keep largest component
    if filter_largest_component:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        #print(Gcc)
        G = G.subgraph(Gcc[0])
        print(G)


    #print(G)
    return G


## PCA/CA
def run_ca_decomposition(graph, topic_no, type_="ca", align_left_user=None, domains=None, transpose=False, bipartite=True):
    """
    Executes a dimensionality reduction technique on the adjacency matrix of a graph.
    
    This function converts a graph into its adjacency matrix representation and then applies either
    Principal Component Analysis (PCA) or Correspondence Analysis (CA) to reduce the dimensionality 
    of the matrix to 'k' dimensions. 
    
    Parameters:
    - graph (networkx.Graph): The graph on which to perform the decomposition.
    - type_ (str): The type of decomposition to perform. Currently, only "ca" for Correspondence Analysis is implemented.
    - k (int): The number of components to return after the decomposition process. Only k=1 is implemented.
    - align_left_user (list|None): align the embedding so the average of these users has a negative embedding
    
    The function performs the following operations:
    - Converts the input graph into a pandas DataFrame adjacency matrix.
    - Depending on the value of 'type_', applies the specified decomposition algorithm.
      If 'type_' is "pca", the function is not implemented and will raise an error.
      If 'type_' is "ca", it fits a Correspondence Analysis model to the adjacency matrix.
    - Prints a summary of eigenvalues after fitting the model.
    - Extracts the coordinates of the columns (nodes) in the reduced space.
    - Aligns the dimensions based on a specific user (presumably active and left-wing) by checking the sign of their embedding.
      If the embedding of this user is positive, the embeddings are multiplied by -1 to align the dimensions accordingly.
    
    Returns:
    - user2emb (pandas.DataFrame): A DataFrame containing the coordinates of the nodes in the reduced k-dimensional space.
    
    Notes:
    - The function is designed with the assumption that there is a specific user with a known identifier that should have
      a positive orientation in the reduced space. This is used to orient the dimensions consistently.
    """
    from sklearn.decomposition import PCA
    from prince import PCA, CA

    #Extract dataframe from graph (useful to keep labels consistent)
    df = nx.to_pandas_adjacency(graph)
    
    if domains:
        if bipartite == True:
            df = df.loc[[_ for _ in df.index if _ in set(domains)], [_ for _ in df.columns if _ not in set(domains)]]
        else:
            df = df.loc[[_ for _ in df.index if _ in set(domains)], :]
    if transpose:
        df = df.T
        
    # PCA
    if type_=="pca":
        raise("Not implemented yet") #note, only need to change .column_coordinates below
        #model = PCA(n_components=k, rescale_with_mean=False, rescale_with_std=False)
    elif type_=="ca":
        model = CA(n_components=50)
        model.fit(df)
        try:
            dimension = np.nonzero(model.cumulative_percentage_of_variance_>50)[0][0]+1
        except IndexError:
            dimension = 50
        
            
        #user2emb = model.column_coordinates(df)[0]
        if bipartite:
            # row_coordinates gives the users
            vectmasked = model.column_coordinates(df)
            vectmasked_row = model.row_coordinates(df)
            # Create embeddings and append to list
            df_embedding = pd.concat([
                pd.DataFrame({"user": vectmasked.index}),
                pd.DataFrame({"user": vectmasked_row.index})
            ])
        else:
            vectmasked = model.column_coordinates(df)
            df_embedding = pd.DataFrame({"user": vectmasked.index})

            # Test correlatoin
            vectmasked_row = model.row_coordinates(df)
            df_embedding_row = pd.DataFrame({"user": vectmasked_row.index})

        for i in range(dimension):
            if bipartite:
                df_embedding[f"{topic_no}_"+str(i)] = np.concatenate([vectmasked.loc[:, i].values, vectmasked_row.loc[:, i].values])
            else:
                df_embedding[f"{topic_no}_"+str(i)] = vectmasked.loc[:, i].values
                df_embedding_row[f"{topic_no}_"+str(i)] = vectmasked_row.loc[:, i].values
                                                       
            # Align dimensions with specific user that is active and left-wing
            if align_left_user is not None:
                users_in_data = list(set(align_left_user) & set(vectmasked.index))
                if len(users_in_data) == 0:
                    print("Careful, none of the users were in the data, the embedding was not reoriented")
                else:
                    if vectmasked.loc[users_in_data].mean() > 0:
                        df_embedding[f"{topic_no}_"+str(i)] *= -1

        if not bipartite:
            # Merge the first 3 numeric cols of the two embeddings to make sure row and columsn are (almost) identical
            cols_merge = list(df_embedding.columns[:4])
            print(cols_merge)
            df_col_row = pd.merge(df_embedding[cols_merge], df_embedding_row[cols_merge], on="user")
            print(df_col_row.head(2))
            # Keep only numeric 
            df_col_row = df_col_row.select_dtypes(include=[np.number])
            print(df_col_row.corr("spearman"))
            
            

    return df_embedding

def plot_annotated_scatter(labels, X, Y, c=None, show=True, adjust=True, default_color="turquoise", s=10, show_title=True, marker="o", alpha=1, cmap=plt.cm.PuOr, textsize=5, textcolor=None):
    if text_color is None:
        textcolor = ["k" if _ != "turquoise" else "darkgray" for _ in c]
    else:
        textcolor = [text_color for _ in c]
    
    if isinstance(c, str):
        c = [c]*len(X)
    elif c is not None:
        c_nonan = c[np.isfinite(c)]
        vmin = min(c_nonan)
        vmax = max(c_nonan)
        
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
        
        c = [cmap(norm(_)) if np.isfinite(_) else default_color for _ in c]
    else:
        c = [palette[0]]*len(X)
        
    for x_,y_,c_ in zip(X,Y,c):
        if c_ == default_color:
            plt.scatter(x_, y_, color=c_, s=s, marker=marker, zorder=2, alpha=alpha)
        else:
            plt.scatter(x_, y_, color=c_, s=s, marker=marker, zorder=3, alpha=alpha)
    # Create a scatter plot, plot

    
    if show_title:
        plt.title(f"Pearson Corr: {pearsonr(X, Y)[0]:2.3f}\nSpearman Corr: {spearmanr(X, Y)[0]:2.3f}")
    
    # if adjust:
    #     texts = []
    #     for lab,x,y,col in zip(labels, X, Y,c):
    #         if col == palette[0]:
    #             texts.append(plt.text(x,y,lab,fontsize=8,color="gray",zorder=6))
    #         elif col != default_color:
    #             texts.append(plt.text(x,y,lab,fontsize=5,color="navy",zorder=9))
    #         else:
    #             texts.append(plt.text(x,y,lab,fontsize=5,color="gray",zorder=5))
    if adjust:
        ta.allocate(plt.gca(),
                    X.values,
                    Y.values,
                    labels.values if isinstance(labels, pd.Series) else labels,
                    textsize=textsize,
                    textcolor=textcolor,
                    linecolor="gray",
                    x_scatter=X,
                    y_scatter=Y,
                    max_distance=0.1)
                

        #adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    
    if show:
        plt.show()
    
def run_sheep_decomposition(graph, topic_no, plot=False, normalize_laplacian=False):
    A2 = nx.to_numpy_array(graph)
    
    if not normalize_laplacian:
        degree = [val for (node, val) in graph.degree(weight = 'weight')]
        D2 = np.diag(degree)
        Lr = D2 - A2

    # Normalized laplacian
    else:
        # 
        A2_pos = A2.copy()
        A2_pos[A2_pos<0] = 0
        A2_neg = -A2.copy()
        A2_neg[A2_neg<0] = 0

        
        degree = A2_pos.sum(1) + A2_neg.sum(1)
        #degree = np.abs([val for (node, val) in graph.degree(weight = 'weight')])
        degree[degree==0] = 1

        # Calculate D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        A_norm =  D_inv_sqrt @ A2 @ D_inv_sqrt
        
        degree = A_norm.sum(1)
        D2 = np.diag(degree)
        
        Lr = D2 - A_norm
        # Normalize the Laplacian matrix
        

    valmasked, vectmasked = np.linalg.eigh(Lr)
    
    if 1:
        #find the minimum of the dimension required 
        k_max = 40
        dims = np.arange(k_max) + 1
        #energy = []
        # Calculate the energy for each dimension in parallel
        with Pool(8) as p:
            energy = p.map(lambda k: energy_repelling_lap_norm_bis(vectmasked[:,0:k], A2, k), dims)
        # for i in range(k_max):
        #     k = i + 1
        #     energy.append(energy_repelling_lap_norm_bis(vectmasked[:,0:k], A2, k))
        if plot:
            plt.plot(dims, energy)
            plt.title(topic_no)
            plt.show()
        
        dim_optimal = np.where(energy == min(energy))[0][0] 
    else:
        dim_optimal = 10
        
    # Create embeddings and append to list
    df_embedding = pd.DataFrame({"user": list(graph.nodes())})
        
    for i in range(dim_optimal+1):
        df_embedding[f"{topic_no}_"+str(i)] = vectmasked[:, i]  
        
    # Create embeddings and append to list
    return df_embedding


def create_embeddings(data, method="sheep", bipartite=True, min_votes_from_user=10, min_votes_to_users_or_domains=0, min_comments_or_stories=10, plot_sheep=True, normalize_laplacian=False, adjust_weight=False, transpose=True):
    # 0, 0 for comparisonwith sheep
    df_all_embeddings = []
    if bipartite == True:
        domains = set(data["story_original_url_domain"].unique())
    else:
        domains = set(data["username_post"].unique())
    # For each topic 
    for topic_no, data_topic in data.groupby("topic"):
        #print(topic_no)
        # Create graph and extract adjacency matrix (other options are also possible)
        graph = meneame_graph_aggregate(data, 
                                        topic_no, 
                                        only_positive_votes=False if method=="sheep" else True, 
                                        filter_largest_component=True, 
                                        min_votes_from_user=min_votes_from_user, 
                                        min_votes_to_users_or_domains=min_votes_to_users_or_domains,
                                        min_comments_or_stories=min_comments_or_stories,
                                        bipartite=bipartite,
                                        adjust_weight=adjust_weight)

        if method == "sheep":
            df_embedding = run_sheep_decomposition(graph, topic_no, plot=plot_sheep, normalize_laplacian=normalize_laplacian)
        elif method == "ca":
            df_embedding = run_ca_decomposition(graph, topic_no, domains=domains, transpose=transpose, bipartite=bipartite)
        else:
            raise(f"Method {method} not implemented")
                                        

        df_all_embeddings.append(df_embedding.set_index("user"))

    # Concatenate embeddings
    df_all_embeddings = pd.concat(df_all_embeddings, axis=1)
    return df_all_embeddings
    
    #df_all_embeddings.to_csv(f"{path_save_embeddings}/embeddings_sheep_bipartite_filtered.csv")
    #df_all_embeddings.corr()


def create_embeddings_third(data, method="sheep", bipartite=True, min_votes_from_user=10, min_votes_to_users_or_domains=0,
                      min_comments_or_stories=10, plot_sheep=True, normalize_laplacian=False, adjust_weight=False,
                      transpose=True, adj_weight = -0.5):
    # 0, 0 for comparisonwith sheep
    df_all_embeddings = []
    if bipartite == True:
        domains = set(data["story_original_url_domain"].unique())
    else:
        domains = set(data["username_post"].unique())
    # For each topic
    for topic_no, data_topic in data.groupby("topic"):
        # print(topic_no)
        # Create graph and extract adjacency matrix (other options are also possible)
        graph = meneame_graph_aggregate(data,
                                        topic_no,
                                        only_positive_votes=False if method == "sheep" else True,
                                        filter_largest_component=True,
                                        min_votes_from_user=min_votes_from_user,
                                        min_votes_to_users_or_domains=min_votes_to_users_or_domains,
                                        min_comments_or_stories=min_comments_or_stories,
                                        bipartite=bipartite,
                                        adjust_weight=adjust_weight)

        if method == "sheep":
            df_embedding = run_sheep_decomposition(graph, topic_no, plot=plot_sheep,
                                                   normalize_laplacian=normalize_laplacian)

            # Get the degrees of all nodes
            weights = [data['weight'] for u, v, data in graph.edges(data=True)]
            # Compute the median
            median_degree = np.median(weights)
            min_weights = np.min(weights)
            print('sheep', median_degree, min_weights)

            num_negative_edges = sum(1 for w in weights if w < 0)
            num_positive_edges = sum(1 for w in weights if w > 0)

            sum_negative_weights = sum(w for w in weights if w < 0)
            sum_positive_weights = sum(w for w in weights if w > 0)

            print("Number of negative edges:", num_negative_edges)
            print("Number of positive edges:", num_positive_edges)
            print("Total sum of negative weights:", sum_negative_weights)
            print("Total sum of positive weights:", sum_positive_weights)

        elif method == 'sheep-adj':

            weights = [data['weight'] for u, v, data in graph.edges(data=True)]
            median_degree = np.median(weights)
            print('sheep-adj_pre', median_degree)

            graph = nx.Graph(graph)
            for node1, node2 in itertools.combinations(graph.nodes(), 2):  # Iterate over all pairs of nodes
                if not graph.has_edge(node1, node2):  # Check if the edge does not exist
                    graph.add_edge(node1, node2, weight=adj_weight)  # Add the edge with the specified weight

            weights = [data['weight'] for u, v, data in graph.edges(data=True)]
            median_degree = np.median(weights)
            print('sheep-adj_post', median_degree)

            df_embedding = run_sheep_decomposition(graph, topic_no, plot=plot_sheep,
                                                   normalize_laplacian=normalize_laplacian)

        elif method == "ca":
            weights = [data['weight'] for u, v, data in graph.edges(data=True)]
            median_degree = np.median(weights)
            print('ca', median_degree)
            df_embedding = run_ca_decomposition(graph, topic_no, domains=domains, transpose=transpose,
                                                bipartite=bipartite)
        else:
            raise (f"Method {method} not implemented")

        df_all_embeddings.append(df_embedding.set_index("user"))

    # Concatenate embeddings
    df_all_embeddings = pd.concat(df_all_embeddings, axis=1)
    return df_all_embeddings

def create_pca_emb(df_all_embeddings, topics, domains=None, normalize=True, emb="sheep"):
    df_dom_embeddings_pca = []
    
    for topic in topics:
        print(topic)
        df_dom_pca = transform_embeddings(df_all_embeddings, topic, domains=domains, normalize=True)
        df_dom_embeddings_pca.append(df_dom_pca)
        
    # Concatenate embeddings
    df_dom_embeddings_pca = pd.concat(df_dom_embeddings_pca, axis=1).reset_index()
    return df_dom_embeddings_pca
    
def compare_with_twitter(df_all_embeddings, topic):
    most_correlated_dimension = 1
    return most_correlated_dimension

def transform_embeddings(df_all_embeddings, topic, domains=None, normalize=True):
    regex_hold = topic + '|story'
    df2 = df_all_embeddings.filter(regex = regex_hold)
    print(topic, "dim: ", len(df2.columns)-1)

    if domains is None:
        df2 = df2[df2['story'].str.len() != 40]
    else:
        df2 = df2.loc[df2["story"].isin(domains)]

    # Dropping
    df2 = df2.dropna()

    df_pca = df2.filter(regex = topic).values

    if normalize:
        scaler = StandardScaler()
        df_pca = scaler.fit_transform(df_pca)
        
    pca = PCA(svd_solver = 'full')
    A = pca.fit_transform(df_pca)
    print(f"Expained variance: {pca.explained_variance_ratio_[:3]}")

    df2[f'{topic}_pca_1d'] = A[:, 0]
    if len(pca.components_) > 1:
        df2[f'{topic}_pca_2d'] = A[:, 1]

    return df2.set_index("story")

def compare_emb_twitter(df_all_embeddings, top, emb="sheep", domains=None, compare_to="twitter", data=None, adjust_labels=True, plot_dist=False, nodes_color=None, path_figure=None):
    if emb in ["sheep", "ca"]:
        suffix1 = "0"
        suffix2 = "1"
    elif emb == "pca":
        suffix1 = "pca_1d"
        suffix2 = "pca_2d"
        
    if f"{top}_1" in df_all_embeddings.columns:
        df_embedding = df_all_embeddings.loc[:, ["story", f"{top}_{suffix1}", f"{top}_{suffix2}"]]
    else: #only 1 sheep dimension
        df_embedding = df_all_embeddings.loc[:, ["story", f"{top}_{suffix1}"]]

    if compare_to == "twitter":
        df_embedding["tw"] = df_embedding["story"].map(domain_ideology_twitter)
        label = "Twitter ideology"
    elif compare_to == "total_votes":
        neg_votes = data.loc[(data["topic"]==top)].groupby("story_original_url_domain")["story_id"].agg(lambda x: len(set(x))).to_dict()
        df_embedding["tw"] = np.log10(1+df_embedding["story"].map(neg_votes))
        label = "Log10 number stories"
    else:
        neg_votes = data.loc[(data["topic"]==top)].groupby("story_original_url_domain")["story_vote"].mean().to_dict()
        df_embedding["tw"] = df_embedding["story"].map(neg_votes)
        label = "Fraction of positive votes"
    #df_embedding["watch"] = df_embedding["story"].map(ideology_media_watch)

    # Remove domains when no Twitter ideology present
    df_embedding_tw = df_embedding.dropna(subset=["tw", f"{top}_{suffix1}"])
    
    
    #df_embedding_w = df_embedding.dropna(subset=["watch", top])
    
    print(top)
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_annotated_scatter(df_embedding_tw["story"], df_embedding_tw[f"{top}_{suffix1}"], df_embedding_tw["tw"], show=False, adjust=adjust_labels)
    plt.xlabel(f"{emb} D1 embedding")
    plt.ylabel(label)

    if f"{top}_{suffix2}" in df_embedding.columns:
        plt.subplot(122)
        plot_annotated_scatter(df_embedding_tw["story"], df_embedding_tw[f"{top}_{suffix2}"], df_embedding_tw["tw"], show=False, adjust=adjust_labels)
        plt.xlabel(f"{emb} D2 embedding")
        plt.ylabel(label)
        
        if path_figure is not None:
            plt.savefig(path_figure)
            plt.figure(figsize=(6,6))
            
        df_embedding = df_embedding.dropna(subset=[f"{top}_{suffix1}", f"{top}_{suffix2}"])
        if domains is None:
            df_embedding_min = df_embedding[df_embedding['story'].str.len() == 40]
            df_embedding = df_embedding[df_embedding['story'].str.len() != 40]
            
            
        else:
            df_embedding_min = df_embedding.loc[~df_embedding["story"].isin(domains)]
            df_embedding = df_embedding.loc[df_embedding["story"].isin(domains)]

        if nodes_color is not None:
            df_embedding_min["color"] = df_embedding_min["story"].map(nodes_color)
            norm = matplotlib.colors.Normalize(vmin = df_embedding_min["color"].min(), vmax = df_embedding_min["color"].max())
            c = [plt.cm.RdBu(norm(_)) if np.isfinite(_) else "turquoise" for _ in df_embedding_min["color"]]
        else:
            c = "forestgreen"
        plt.scatter(df_embedding_min[f"{top}_{suffix1}"], df_embedding_min[f"{top}_{suffix2}"], c=c, s=1)
        plot_annotated_scatter(df_embedding["story"], df_embedding[f"{top}_{suffix1}"], df_embedding[f"{top}_{suffix2}"], c=df_embedding["tw"], show=False, adjust=False)
        
        
        plt.xlabel(f"{emb} D1 embedding")
        plt.ylabel(f"{emb} D2 embedding")

    plt.show()
    if plot_dist:
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        sns.kdeplot(x=f"{top}_{suffix1}", data = df_embedding)#, bins=50)
    
        if f"{top}_{suffix2}" in df_embedding.columns:
            plt.subplot(122)
            sns.kdeplot(x=f"{top}_{suffix2}", data = df_embedding)#, bins=50)
        plt.show()
