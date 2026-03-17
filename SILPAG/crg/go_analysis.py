"""
Pathway Enrichment Analysis and Visualization for SILPAG-identified CRGs.

Uses gseapy (Enrichr) to run pathway enrichment (KEGG, Reactome, GO, etc.),
then produces a publication-ready bubble plot.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def run_pathway_enrichment(
    gene_list,
    gene_sets='KEGG_2021_Human',
    organism='human',
    top_n=10,
):
    """
    Run pathway enrichment via gseapy/Enrichr.

    Parameters
    ----------
    gene_list : list[str]
        Gene symbols to test.
    gene_sets : str
        Enrichr library name. Common choices:
        - 'KEGG_2021_Human'
        - 'Reactome_2022'
        - 'GO_Biological_Process_2021'
    organism : str
        Organism for Enrichr query.
    top_n : int
        Number of top terms to keep (by adjusted p-value).

    Returns
    -------
    df : pd.DataFrame
    """
    import gseapy as gp

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        no_plot=True,
    )
    df = enr.results.copy()

    # Parse overlap column "k/n" -> Gene Count, Gene Ratio
    overlap_split = df['Overlap'].str.split('/', expand=True).astype(int)
    df['Gene Count'] = overlap_split[0]
    df['Gene Ratio'] = overlap_split[0] / overlap_split[1]
    df['-log10(P)'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-300))

    # Sort by adjusted p-value, keep top_n
    df = df.sort_values('Adjusted P-value').head(top_n).reset_index(drop=True)

    # Clean term names, wrap long text
    df['Term'] = df['Term'].apply(lambda x: re.sub(r'\s*\(GO:\d+\)', '', x))
    df['Term'] = df['Term'].apply(lambda x: _wrap_text(x, max_width=40))

    return df


# Keep old name as alias for backward compatibility
run_go_enrichment = run_pathway_enrichment


def _wrap_text(text, max_width=40):
    """Wrap long text at word boundaries for y-axis labels."""
    words = text.split()
    lines, current_line, current_len = [], [], 0
    for w in words:
        if current_len + len(w) + 1 > max_width and current_line:
            lines.append(' '.join(current_line))
            current_line, current_len = [w], len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1
    if current_line:
        lines.append(' '.join(current_line))
    return '\n'.join(lines)


def plot_pathway_bubble(
    df,
    size_scale=15,
    figsize=(12, 6.5),
    cmap='coolwarm',
    title='Pathway Enrichment',
    save_path=None,
):
    """
    Publication-ready bubble plot for pathway enrichment results.
    """
    fig, ax = plt.subplots(figsize=figsize)

    bubble = ax.scatter(
        df['Gene Ratio'],
        df['Term'],
        s=[c * size_scale for c in df['Gene Count']],
        c=df['-log10(P)'],
        cmap=cmap,
        edgecolor='k',
        alpha=0.7,
    )

    # Colorbar
    cbar = plt.colorbar(bubble, ax=ax)
    cbar.set_label('-log10(P.adjust)', fontsize=18)
    cbar.ax.tick_params(labelsize=15, length=6, width=2)
    cbar.outline.set_linewidth(2)

    # Size legend
    counts = df['Gene Count']
    size_legend_vals = [int(np.percentile(counts, 25)), int(np.percentile(counts, 75))]
    size_legend_vals = sorted(set(max(v, 1) for v in size_legend_vals))
    for s in size_legend_vals:
        ax.scatter([], [], s=s * size_scale, color='gray', alpha=0.6, label=f'{s} genes')
    ax.legend(
        scatterpoints=1, frameon=True, labelspacing=0.9,
        title='Gene Count', fontsize=14, title_fontsize=14,
        borderpad=0.8, edgecolor='black', framealpha=0.8, borderaxespad=0.7,
    )

    ax.set_xlabel('Gene Ratio', fontsize=20)
    ax.set_title(title, fontsize=22, pad=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='both', which='major', length=6, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(False)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# Keep old name as alias
plot_go_bubble = plot_pathway_bubble
