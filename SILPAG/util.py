import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from dynamicTreeCut import cutreeHybrid
from scipy.stats import fisher_exact
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import anndata2ri
from rpy2.robjects import numpy2ri, r, globalenv
from rpy2.robjects.packages import importr


def cluster_leiden(adata: ad.AnnData,n_cluster: int,random_state=None):
    # sc.pp.neighbors(adata,random_state=random_state)
    r = 0.5
    r1 = 1
    k = 0
    while(k!=n_cluster):
        sc.tl.leiden(adata, resolution=r)
        k = adata.obs['leiden'].nunique()
        if k>n_cluster:
            r = r/2
            r1 = r1/2
        elif k<n_cluster:
            r = (r+r1)/2
            r1 = r+r1
    return adata.obs['leiden']


def SPARKX_cluster(adata, threshold=0.01, h=0.2): 
    """
    SPARKX + Hierarchical Clustering
    """
    anndata2ri.activate()
    numpy2ri.activate()
    
    # process data
    matrix_py = adata.X.T
    if sparse.issparse(matrix_py):
        matrix_py = matrix_py.toarray()
    matrix_py = pd.DataFrame(
        data = matrix_py,
        index = adata.var_names,
        columns = adata.obs_names
    )
    location = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)

    # run SPARKX
    importr("SPARK")
    globalenv['matrix_py'] = matrix_py
    globalenv['location'] = location
    r("""
    matrix_py <- as.matrix(matrix_py)
    location <- as.matrix(location)

    sparkX <- sparkx(matrix_py, location, numCores=1, option="mixture")

    spark_results <- sparkX$res_mtest
    """)
    adata.var['SPARKX'] = r('spark_results')['adjustedPval']
    adata_filt = adata[:, adata.var['SPARKX'] <= threshold].copy()

    # run Hierarchical Clustering
    matrix_filt = adata_filt.X.T
    if sparse.issparse(matrix_filt):
        matrix_filt = matrix_filt.toarray()

    matrix_filt = pd.DataFrame(
        data = matrix_filt,
        index = adata_filt.var_names,
        columns = adata_filt.obs_names
    )

    importr("amap")
    globalenv['matrix_filt'] = matrix_filt
    globalenv['h'] = h
    r("""
    matrix_filt <- as.matrix(matrix_filt)

    res_hc <- hcluster(x = matrix_filt, 
                    method = "correlation", 
                    link = "average", 
                    nbproc = 1)
    cluster_ids <- cutree(res_hc, h = h)
    """)
    adata_filt.var['cluster'] = r('cluster_ids')
    return adata, adata_filt


def run_SPARKX(adata): 
    """
    SPARKX
    """
    anndata2ri.activate()
    numpy2ri.activate()
    
    # process data
    matrix_py = adata.X.T
    if sparse.issparse(matrix_py):
        matrix_py = matrix_py.toarray()
    matrix_py = pd.DataFrame(
        data = matrix_py,
        index = adata.var_names,
        columns = adata.obs_names
    )
    location = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)

    # run SPARKX
    importr("SPARK")
    globalenv['matrix_py'] = matrix_py
    globalenv['location'] = location
    r("""
    matrix_py <- as.matrix(matrix_py)
    location <- as.matrix(location)

    sparkX <- sparkx(matrix_py, location, numCores=1, option="mixture")

    spark_results <- sparkX$res_mtest
    """)
    # adata.var['SPARKX'] = r('spark_results')['adjustedPval']
    return r('spark_results')['adjustedPval']


def run_dynamicTreeCut(adata, minClusterSize=20, deepSplit=1): 
    """
    Hierarchical Clustering + dynamicTreeCut
    """

    matrix_py = adata.X.T
    if sparse.issparse(matrix_py):
        matrix_py = matrix_py.toarray()

    dists = pdist(matrix_py, metric="correlation")
    link_matrix = linkage(dists, method="average")
    clusters = cutreeHybrid(link_matrix, dists, minClusterSize=minClusterSize, deepSplit=deepSplit)

    return clusters['labels']


def spatial_smooth_expression(
    adata: ad.AnnData,
    gene_name: str,
    bandwidth: float = None,
    n_neighbors: int = 20,
    use_knn: bool = None,
    key_added: str = None
) -> ad.AnnData:
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import kneighbors_graph
    if 'spatial' not in adata.obsm:
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
    
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in adata.var_names")
    
    if key_added is None:
        key_added = f"{gene_name}_smoothed"
    
    coords = adata.obsm['spatial']
    n_obs = coords.shape[0]
    expr = adata[:, gene_name].X.toarray().flatten()
    
    if use_knn is None:
        use_knn = n_obs > 3000 
    
    if bandwidth is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, n_obs-1)).fit(coords)
        distances, _ = nn.kneighbors()
        bandwidth = np.median(distances[:, 1:]) * 2.5
        print(f"Auto-calculated bandwidth: {bandwidth:.2f}")
    
    if use_knn:
        knn_graph = kneighbors_graph(
            coords, 
            n_neighbors=n_neighbors+1,
            mode='distance',
            include_self=True
        )
        weights_data = np.exp(-knn_graph.data**2 / (2 * bandwidth**2))
        weights_matrix = knn_graph.copy()
        weights_matrix.data = weights_data
        row_sums = np.array(weights_matrix.sum(axis=1)).flatten()
        weights_matrix.data /= row_sums[weights_matrix.nonzero()[0]]
        smoothed = weights_matrix.dot(expr)
    else:
        distances = pairwise_distances(coords)
        weights = np.exp(-distances**2 / (2 * bandwidth**2))
        np.fill_diagonal(weights, 0) 
        
        row_sums = weights.sum(axis=1)
        weights /= row_sums[:, np.newaxis]
        
        smoothed = weights.dot(expr)

    adata.obs[key_added] = smoothed
    
    return adata


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)


def compute_fisher_p_values(z1, z2, k=15, metric='cosine'):

    total_genes = z1.shape[0]
    
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("z1 and z2 must have the same number of genes.")
    
    def get_knn_neighbors(embeddings, k, metric):
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)
        return {i: set(indices[i][1:]) for i in range(len(embeddings))}

    neighbors_1 = get_knn_neighbors(z1, k, metric)
    neighbors_2 = get_knn_neighbors(z2, k, metric)

    p_values = np.zeros(total_genes)

    for gene in range(total_genes):

        neighbors_1_gene = neighbors_1[gene]
        neighbors_2_gene = neighbors_2[gene]

        a = len(neighbors_1_gene & neighbors_2_gene) 
        b = len(neighbors_1_gene - neighbors_2_gene)
        c = len(neighbors_2_gene - neighbors_1_gene)
        d = total_genes - (a + b + c)

        contingency_table = np.array([[a, b], [c, d]])
        
        _, p_value = fisher_exact(contingency_table, alternative="greater")
        
        p_values[gene] = p_value

    return p_values


def pad_to_divisible(image_array, patch_size=4):
    """
    Pads the last two dimensions of the input image array with zeros if they are not divisible by a specified patch size.

    Parameters:
    - image_array: A NumPy array with at least two dimensions, where the last two dimensions
                   represent the height (H) and width (W) of the image(s).
    - patch_size: An integer specifying the size for divisibility. Default is 4.

    Returns:
    - padded_image: A new NumPy array where H and W are padded to be divisible by patch_size.
    """
    # Get the current height and width
    H, W = image_array.shape[-2], image_array.shape[-1]

    # Calculate the padding needed for height and width
    pad_height = (patch_size - H % patch_size) if H % patch_size != 0 else 0
    pad_width = (patch_size - W % patch_size) if W % patch_size != 0 else 0

    # Apply padding only if needed
    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(
            image_array, 
            pad_width=((0, 0),) * (image_array.ndim - 2) + ((0, pad_height), (0, pad_width)), 
            mode='constant', 
            constant_values=0
        )
    else:
        padded_image = image_array  # No padding needed if H and W are already divisible by patch_size

    return padded_image

def unpad_to_original(padded_image, original_shape):
    """
    Removes the padding from the padded image array to restore it to the original shape.

    Parameters:
    - padded_image: A NumPy array that has been padded.
    - original_shape: A tuple containing the original height and width (H, W) before padding.

    Returns:
    - unpadded_image: The image array restored to its original shape.
    """
    _, H, W = original_shape
    # Slice the array to remove the padding
    unpadded_image = padded_image[..., :H, :W]
    return unpadded_image

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


