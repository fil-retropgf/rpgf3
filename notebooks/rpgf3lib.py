import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm
import itertools
import random
import plotly.graph_objects as go

def get_allocations(df, quorum_cutoff=5, min_funding=250, total_funding=270000, scoring_fn='sum'):
    """
    Calculate project allocations with iterative normalization.
    
    Args:
        df: DataFrame with columns 'projectId' and 'amount'
        quorum_cutoff: Minimum number of votes required
        min_funding: Minimum funding amount required
        total_funding: Total funding to distribute
        scoring_fn: How to aggregate votes ('sum', 'mean', 'median')
    
    Returns:
        tuple: (
            dict of final allocations,
            set of all eliminated projects,
            dict of elimination reasons,
            dict of voting statistics per project
        )
    """
    # Track all eliminated projects and their reasons
    eliminated_projects = set()
    elimination_reasons = {}
    
    # Track voting statistics for all projects
    voting_stats = {}
    
    # Initialize valid projects as all projects
    valid_projects = set(df['projectId'].unique())
    
    # Calculate initial voting statistics for all projects
    for project_id in valid_projects:
        project_votes = df[df['projectId'] == project_id]
        
        # Calculate score based on scoring function
        if scoring_fn == 'sum':
            score = project_votes['amount'].sum()
        elif scoring_fn == 'mean':
            score = project_votes['amount'].mean()
        elif scoring_fn == 'median':
            score = project_votes['amount'].median()
        else:
            raise ValueError(f"Unknown scoring function: {scoring_fn}")
            
        voting_stats[project_id] = {
            'num_votes': len(project_votes),
            'score': score,
            'voters': set(project_votes['voterId']),
            'elimination_round': None,
            'raw_allocation': None,
            'normalized_allocation': None,
            'category': project_votes['category'].iloc[0] if 'category' in project_votes.columns else 'Uncategorized'
        }
    
    round_number = 0
    
    while True:
        round_number += 1
        # Work only with votes for valid projects
        valid_df = df[df['projectId'].isin(valid_projects)]
        
        # Count votes per project
        vote_counts = valid_df.groupby('projectId').size()
        
        # Check quorum
        failed_quorum = set(vote_counts[vote_counts < quorum_cutoff].index)
        for proj in failed_quorum:
            elimination_reasons[proj] = f"Failed quorum with {vote_counts[proj]} votes"
            voting_stats[proj]['elimination_round'] = round_number
        eliminated_projects.update(failed_quorum)
        
        # Calculate vote-weighted allocations for remaining projects
        if len(valid_projects - failed_quorum) == 0:
            return {}, eliminated_projects, elimination_reasons, voting_stats
            
        valid_df = valid_df[~valid_df['projectId'].isin(failed_quorum)]
        
        # Calculate scores based on scoring function
        if scoring_fn == 'sum':
            scores = valid_df.groupby('projectId')['amount'].sum()
        elif scoring_fn == 'mean':
            scores = valid_df.groupby('projectId')['amount'].mean()
        elif scoring_fn == 'median':
            scores = valid_df.groupby('projectId')['amount'].median()
            
        total_score = scores.sum()

        # print(f'Round {round_number} total score: {total_score}')
        
        # Calculate normalized allocations
        allocations = {}
        for proj_id, score in scores.items():
            normalized_amount = score / total_score * total_funding
            allocations[proj_id] = normalized_amount
            
            # Update voting stats
            voting_stats[proj_id].update({
                'raw_allocation': score,
                'normalized_allocation': normalized_amount,
            })
        
        # Check minimum funding
        failed_min_funding = {
            proj_id for proj_id, alloc in allocations.items()
            if alloc < min_funding
        }
        for proj in failed_min_funding:
            elimination_reasons[proj] = f"Failed minimum funding with {allocations[proj]:.2f}"
            voting_stats[proj]['elimination_round'] = round_number
        
        # Combine all newly eliminated projects
        newly_eliminated = failed_quorum | failed_min_funding
        
        # If no new projects were eliminated, we're done
        if not newly_eliminated:
            # Update final allocations in voting stats for successful projects
            for proj_id in valid_projects:
                voting_stats[proj_id]['final_status'] = 'funded'
                voting_stats[proj_id]['final_allocation'] = allocations[proj_id]
            
            # Mark eliminated projects
            for proj_id in eliminated_projects:
                voting_stats[proj_id]['final_status'] = 'eliminated'
                voting_stats[proj_id]['final_allocation'] = 0
                
            return allocations, eliminated_projects, elimination_reasons, voting_stats
            
        # Update tracking sets
        eliminated_projects.update(newly_eliminated)
        valid_projects -= newly_eliminated

def create_failed_projects_df(eliminated, reasons, stats, project_names=None):
    """
    Create a DataFrame of failed projects and their voting statistics.
    """
    if not eliminated:
        return pd.DataFrame()
    
    data = []
    for proj_id in eliminated:
        project_stats = stats[proj_id]
        row = {
            'Project Name': project_names.get(proj_id, "") if project_names else "",
            'Vote Count': project_stats['num_votes'],
            'Score': project_stats['score'],
            'Round': project_stats['elimination_round'],
            'Reason': reasons[proj_id],
            'category': project_stats.get('category', 'Uncategorized')  # Add category with default
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    if not df.empty:
        # Sort by elimination round
        df = df.sort_values('Round')
    
    return df

def create_successful_projects_df(df, allocations, project_names=None):
    """
    Create a DataFrame of successful allocations with statistics.
    """
    data = []
    for proj_id, amount in allocations.items():
        project_votes = df[df['projectId'] == proj_id]['amount']
        row = {
            # 'Project ID': proj_id,
            'Project Name': project_names.get(proj_id, "") if project_names else "",
            'Vote Count': len(project_votes),
            'Average Score': project_votes.mean(),
            'Std Dev': project_votes.std(),
            'Final Allocation (FIL)': amount,
            'category': df[df['projectId'] == proj_id]['category'].iloc[0] if 'category' in df.columns else "Uncategorized"
        }
        data.append(row)
    
    df_success = pd.DataFrame(data)
    if not df_success.empty:
        # Sort by Project ID
        df_success = df_success.sort_values('Final Allocation (FIL)', ascending=False)
        # Format currency columns
        currency_cols = ['Average Vote', 'Std Dev', 'Final Allocation (FIL)']
        # for col in currency_cols:
        #     df_success[col] = df_success[col].map('${:,.2f}'.format)
    
    return df_success

def run_allocation_analysis(df, quorum_cutoff=5, min_funding=250, total_funding=270000, project_names=None, scoring_fn='sum', verbose=False):
    """
    Run the full allocation analysis and return pandas DataFrames for reporting.
    
    Args:
        df: DataFrame with voting data
        quorum_cutoff: Minimum number of votes required
        min_funding: Minimum funding amount required
        total_funding: Total funding to distribute
        project_names: Optional dictionary mapping project IDs to names
    
    Returns:
        tuple: (allocations, eliminated, reasons, stats, failed_df, success_df)
    """
    # Get allocation results
    allocations, eliminated, reasons, stats = get_allocations(
        df, quorum_cutoff, min_funding, total_funding, scoring_fn
    )
    
    # Create DataFrames for display
    failed_df = create_failed_projects_df(eliminated, reasons, stats, project_names)
    success_df = create_successful_projects_df(df, allocations, project_names)
        
    # Display results
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', None)
    
    if verbose:
        print("FAILED PROJECTS REPORT")
        print("=====================")
        if failed_df.empty:
            print("No projects were eliminated.")
        else:
            print(failed_df.to_string(index=False))
    
    if verbose:
        print("\nSUCCESSFUL ALLOCATIONS SUMMARY")
        print("==============================")
        if success_df.empty:
            print("No projects were successful.")
        else:
            print(success_df.to_string(index=False))
    
    # Print funding summary
    total_allocated = sum(allocations.values())
    if verbose:
        print(f"\nFUNDING SUMMARY")
        print("===============")
        print(f"Total Allocated: ${total_allocated:,.2f}")
        print(f"Total Available: ${total_funding:,.2f}")
        print(f"Remaining: ${(total_funding - total_allocated):,.2f}")
    
    return allocations, eliminated, reasons, stats, failed_df, success_df

def votes_distribution_stem(df, figsize=(8, 5), quorum_cutoff=5, save_fp=None):
    gb = df.groupby('projectId')
    project2numvotes = {}
    for k, v, in gb:
        project2numvotes[k] = len(v)

    votes_list = np.asarray(list(project2numvotes.values()))    

    # distribution of votes per project
    fig, axx = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    bins = np.arange(0,votes_list.max()+1.5)-0.5

    ax = axx[0]
    hist, bin_edges = np.histogram(votes_list, bins=bins, density=True)
    (markerline, stemlines, baseline) = ax.stem(bin_edges[:-1]+0.5, hist, linefmt='gray', markerfmt='o', basefmt="k-")
    plt.setp(markerline, 'markerfacecolor', 'teal')  # Change marker face color
    plt.setp(stemlines, 'color', 'teal', 'linewidth', 1.5)  # Change stem line color and increase line width
    # ax.hist(votes_list,bins, density=True, color='skyblue', edgecolor='black')
    # ax.axvline(5, color='red', linestyle='dashed', linewidth=2)  # Adds a vertical line at x=5
    # ax.text(5.25, 0.95 * max(n), 'Quorum Cutoff', rotation=0, verticalalignment='center', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    ax.set_title('Density Histogram', fontsize=14)  # Adding a title
    ax.set_xlabel('Ballots cast per Project', fontsize=12)  # X-axis label
    ax.set_ylabel('Probability', fontsize=12)  # Y-axis label
    # ax.set_xlim([0, 20])

    ax = axx[1]
    # n, bins, patches = ax.hist(votes_list,bins, density=True, cumulative=True, color='lightgreen', edgecolor='black')
    hist, bin_edges = np.histogram(votes_list, bins=bins, density=True)
    cdf = np.cumsum(hist)
    (markerline, stemlines, baseline) = ax.stem(bin_edges[:-1]+0.5, cdf, linefmt='gray', markerfmt='o', basefmt="k-")
    plt.setp(markerline, 'markerfacecolor', 'teal')  # Change marker face color
    plt.setp(stemlines, 'color', 'teal', 'linewidth', 1.5)  # Change stem line color and increase line width
    ax.axvline(quorum_cutoff, color='red', linestyle='dashed', linewidth=2,  alpha=0.5)  # Adds a vertical line at x=5
    ax.text(5.25, 0.95, 'Quorum Cutoff', rotation=0, verticalalignment='center', fontsize=12, color='red',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    ax.set_title('Cumulative Distribution Function', fontsize=14)  # Adding a title
    ax.set_xlabel('Ballots cast per Project', fontsize=12)  # X-axis label
    ax.set_ylabel('Cumulative Probability', fontsize=12)  # Y-axis label
    # ax.set_xlim([0, 20])
    
    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp)


def votes_per_project_histogram(df, figsize=(8, 5), save_fp=None):
    plt.figure(figsize=figsize)
    vc = df['amount'].value_counts()

    # plot the histogram
    plt.hist(df['amount'], bins=np.arange(0, df['amount'].max()+100, 100), density=False, alpha=0.75, color='skyblue', edgecolor='black')
    plt.xlabel('FIL allocated to a given project', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)  # Adding a y-label for clarity
    plt.title('Badgeholder vote allocations', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp)

def project_patterns_withname(df, figsize=(12, 8), save_fp=None, quorum_threshold=6):
    ## visualize heatmap of array of votes, ordered in some way
    # each row is a badgeholder
    # each column is their vote.  ragged array but we can pad with something
    gb = df.groupby('project')
    project2votes = {}
    max_len = 0
    for k, v in gb:
        vv = v['amount'].values
        project2votes[k] = vv
        max_len = max(max_len, len(vv))

    num_voters = len(gb)
    filled_array = np.zeros((num_voters, max_len))
    # now create a non-ragged array
    project_list = []
    for i, (k, v) in enumerate(project2votes.items()):
        filled_array[i, :len(v)] = v
        filled_array[i, len(v):] = np.nan

        # Improved project name truncation with ellipsis
        project_str = k[0:25] + '...' if len(k) > 25 else k
        project_list.append(project_str)

    # sort by # of nans
    nan_counts = np.sum(np.isnan(filled_array), axis=1)
    sorted_indices = np.argsort(nan_counts)
    filled_array = filled_array[sorted_indices]
    project_list = [project_list[i] for i in sorted_indices]

    # Calculate number of votes per project for quorum check
    votes_per_project = np.sum(~np.isnan(filled_array), axis=1)
    below_quorum_mask = votes_per_project < quorum_threshold

    # order the array more nicely
    filled_array1 = np.where(np.isnan(filled_array), -100, filled_array)
    ix_resort = np.argsort(filled_array1, axis=1)[:,::-1]
    sorted_filled_array = np.zeros_like(filled_array1)
    for i in range(len(ix_resort)):
        sorted_filled_array[i] = filled_array1[i, ix_resort[i]]
    sorted_filled_array[sorted_filled_array == -100] = np.nan

    # Creating a custom colormap with better contrast
    original_cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
    colors = original_cmap(np.linspace(0, 1, 256))
    custom_cmap = ListedColormap(colors)

    # Create formatted annotation array
    def format_value(val):
        if np.isnan(val):
            return ''
        if val >= 1e6:
            return f'{val/1e6:.1f}M'
        elif val >= 1e3:
            return f'{val/1e3:.1f}K'
        else:
            return f'{val:.0f}'

    annot = np.empty_like(sorted_filled_array, dtype='object')
    for i in range(sorted_filled_array.shape[0]):
        for j in range(sorted_filled_array.shape[1]):
            annot[i,j] = format_value(sorted_filled_array[i,j])

    # Create figure with more space on right for colorbar
    plt.figure(figsize=figsize)
    
    # Create thinner colorbar
    cbar_kws = {
        'label': 'Vote Amount',
        'fraction': 0.02,  # Make colorbar thinner
        'pad': 0.04  # Adjust spacing between heatmap and colorbar
    }
    
    ax = sns.heatmap(sorted_filled_array,
                     annot=annot,
                     fmt='',
                     annot_kws={"size": 8},
                     cmap=custom_cmap,
                     linewidths=0.5,
                     cbar_kws=cbar_kws,
                     yticklabels=project_list,
                     xticklabels=range(1, max_len+1))

    # Add horizontal lines to separate quorum/non-quorum projects
    quorum_transitions = np.where(np.diff(below_quorum_mask))[0]
    if len(quorum_transitions) > 0:
        last_quorum_project = quorum_transitions[0] + 1
        plt.axhline(y=last_quorum_project, color='red', linestyle='--', alpha=0.7)
        
        # Add text annotation for quorum line in the middle of the graph
        mid_x = max_len / 2
        plt.text(mid_x, last_quorum_project + 0.2,  # Slightly above the line
                f'Quorum Cutoff ({quorum_threshold} votes)', 
                horizontalalignment='center',
                verticalalignment='bottom',
                color='red',
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))  # White background for better visibility

    # Modify y-tick labels color individually
    for idx, tick in enumerate(ax.yaxis.get_ticklabels()):
        if below_quorum_mask[idx]:
            tick.set_color('darkgray')
        else:
            tick.set_color('black')

    plt.xlabel('Vote Number', fontsize=12)
    plt.ylabel('Project', fontsize=12)
    plt.title('Project Voting Patterns', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent text overlap
    plt.tight_layout()

    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight', dpi=300)

def badgeholder_patterns2(df, figsize=(12, 8), save_fp=None):
    ## visualize heatmap of array of votes, ordered in some way
    # each row is a badgeholder
    # each column is their vote.  ragged array but we can pad with something
    gb = df.groupby('voterId')
    voter2votes = {}
    max_len = 0
    for k, v in gb:
        vv = v['amount'].values
        voter2votes[k] = vv
        max_len = max(max_len, len(vv))

    num_voters = len(gb)
    filled_array = np.zeros((num_voters, max_len))
    # now create a non-ragged array
    original_voter_ids = []
    for i, (k, v) in enumerate(voter2votes.items()):
        filled_array[i, :len(v)] = v
        filled_array[i, len(v):] = np.nan
        original_voter_ids.append(k)

    # sort by # of nans
    nan_counts = np.sum(np.isnan(filled_array), axis=1)
    sorted_indices = np.argsort(nan_counts)
    filled_array = filled_array[sorted_indices]
    
    # Create voter labels after sorting - label AI Badgeholder explicitly
    sorted_voter_ids = [original_voter_ids[i] for i in sorted_indices]
    voter_list = []
    counter = 1
    for voter_id in sorted_voter_ids:
        if voter_id == 'AI Badgeholder':
            voter_list.append('AI')
        else:
            voter_list.append(f"{counter}")
            counter += 1

    # order the array more nicely
    filled_array1 = np.where(np.isnan(filled_array), -100, filled_array)
    sorted_filled_array = np.sort(filled_array1, axis=1)[:,::-1]
    sorted_filled_array[sorted_filled_array == -100] = np.nan

    # Creating a custom colormap with better contrast
    original_cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
    colors = original_cmap(np.linspace(0, 1, 256))
    custom_cmap = ListedColormap(colors)

    # Create figure
    plt.figure(figsize=figsize)
    
    # Create thinner colorbar with formatted labels
    def format_value(val, pos):
        if val >= 1e6:
            return f'{val/1e6:.1f}M'
        elif val >= 1e3:
            return f'{val/1e3:.1f}K'
        else:
            return f'{val:.0f}'
            
    cbar_kws = {
        'label': 'Vote Amount',
        'fraction': 0.02,  # Make colorbar thinner
        'pad': 0.04,  # Adjust spacing between heatmap and colorbar
        'format': FuncFormatter(format_value)  # Format colorbar labels
    }
    
    # Calculate voting statistics
    votes_per_badgeholder = np.sum(~np.isnan(filled_array), axis=1)
    median_votes = np.median(votes_per_badgeholder)
    mean_votes = np.mean(votes_per_badgeholder)
    
    ax = sns.heatmap(sorted_filled_array,
                     annot=False,  # Remove cell annotations
                     cmap=custom_cmap,
                     linewidths=0.5,
                     cbar_kws=cbar_kws,
                     yticklabels=voter_list,
                     xticklabels=range(1, max_len+1))

    # Make the AI Badgeholder label bold and colored
    yticklabels = ax.get_yticklabels()
    for i, label in enumerate(yticklabels):
        if label.get_text() == 'AI':
            label.set_fontweight('bold')
            label.set_color('#2d5016')  # Dark green color to make it stand out
            label.set_fontsize(label.get_fontsize() * 1.3)  # Larger font size

    # Add statistics text in the middle of the graph
    mid_x = max_len / 2
    stats_text = f'Median votes per badgeholder: {median_votes:.1f}\nMean votes per badgeholder: {mean_votes:.1f}'
    plt.text(mid_x, num_voters - 2, 
            stats_text,
            horizontalalignment='center',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.xlabel('Vote Number', fontsize=12)
    plt.ylabel('Badgeholder', fontsize=12)
    plt.title('Badgeholder Voting Patterns', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent text overlap
    plt.tight_layout()

    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight', dpi=300)

def project_distribution_patterns(allocations, figsize=(9,4), save_fp=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,4))

    data=np.sort(np.asarray(list(allocations.values())))[::-1]
    
    axx = ax[0]
    axx.plot(np.arange(len(data)), data)
    # axx.legend(title='Scoring Configuration')
    axx.set_xlabel('Project Index')
    axx.set_ylabel('Allocation (FIL)')

    axx = ax[1]
    total_fil = np.sum(data)
    xx = data.cumsum()/total_fil * 100
    axx.plot(np.arange(len(data)), xx)

    pct = [10, 25, 50, 75, 100]
    xticks = []
    xticklabs = []
    yticklabs = []
    for p in pct:
        # find the closest index to this pct
        ix = np.argmin(np.abs(xx - p))
        # print(p, ix)

        axx.vlines(ix, 0, p, color='r', linestyle='--')
        axx.hlines(p, 0, ix, color='r', linestyle='--')
        # axx.text(ix, p, f'{p}%', color='r', fontsize=10, ha='right')
        # put a label on the axis for this pct
        # axx.text(ix, 0, f'{ix}', color='k', fontsize=10, va='bottom')
        xticks.append(ix)
        xticklabs.append('%0.02f' % ((ix+1)/len(data) * 100,))
        yticklabs.append(p)

    # axx.legend(title='Scoring Configuration')
    axx.set_ylim(0, 105)
    axx.set_xlim(0, len(data))
    # axx.set_xticklabels(xticklabs)
    axx.set_xticks(xticks)
    axx.set_xticklabels(xticklabs, rotation=45)
    axx.set_yticks(yticklabs)
    axx.set_xlabel('Project Rank (Pct)')
    axx.set_ylabel('% Total Allocation')

    plt.suptitle('Project Allocation Distribution')

    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp)

def get_multiple_groups(groupby_obj, voter_indices):
    """Get data for specified voters"""
    unique_voters = list(groupby_obj.groups.keys())
    selected_voters = [unique_voters[i] for i in voter_indices]
    group_slices = []
    
    for voter in selected_voters:
        try:
            group_slices.append(groupby_obj.get_group(voter))
        except KeyError:
            pass
    return pd.concat(group_slices) if group_slices else pd.DataFrame()

def bootstrap_allocation_distribution(
    df, 
    n_samples=100,
    min_voters=25, 
    quorum_cutoff=6, 
    min_funding=250, 
    total_funding=270000,
    projectid2name=None,
    sample_size=None
):
    """
    Bootstrap analysis of allocation distributions by sampling different voter subsets
    
    Parameters:
    - df: DataFrame with columns [voterId, projectId, amount]
    - n_samples: Number of bootstrap samples to generate
    - min_voters: Minimum number of voters in each sample
    - sample_size: If None, will randomly choose between min_voters and total voters
    """
    gb = df.groupby('voterId')
    num_total_voters = len(gb)
    
    all_allocations = []
    rng = np.random.default_rng()
    
    for _ in tqdm(range(n_samples)):
        # Randomly sample voters
        n_voters = sample_size if sample_size else random.randint(min_voters, num_total_voters)
        # voter_indices = random.sample(range(num_total_voters), n_voters)
        voter_indices = rng.choice(range(num_total_voters), n_voters)
        
        # Get data for sampled voters
        sample_df = get_multiple_groups(gb, voter_indices)
        
        # Calculate allocation for this sample
        allocations, eliminated, reasons, stats, failed_df, success_df = run_allocation_analysis(
            sample_df,
            quorum_cutoff=quorum_cutoff,
            min_funding=min_funding,
            total_funding=total_funding,
            project_names=projectid2name,
            verbose=False
        )
        allocation_sorted = np.sort(np.asarray(list(allocations.values())))[::-1]
        all_allocations.append(allocation_sorted)
    
    return all_allocations

def pad_and_stack_arrays(array_list):
    # Input validation
    if not array_list:
        raise ValueError("Input list cannot be empty")
    
    if not all(isinstance(arr, np.ndarray) for arr in array_list):
        raise TypeError("All elements must be numpy arrays")
        
    if not all(arr.ndim == 1 for arr in array_list):
        raise ValueError("All arrays must be 1-dimensional")
    
    # Find the length of the longest array
    max_length = max(len(arr) for arr in array_list)
    
    # Create padded arrays
    padded_arrays = []
    for arr in array_list:
        padding_length = max_length - len(arr)
        padded_arr = np.pad(arr, (0, padding_length), mode='constant', constant_values=0)
        padded_arrays.append(padded_arr)
    
    # Stack the padded arrays into a matrix
    return np.vstack(padded_arrays)

# plot the distribution of the bootstrap samples
def plot_bootstrap_distribution(bootstrap_allocations, actual_allocation, figsize=(9,4), save_fp=None):
    data_matrix = pad_and_stack_arrays(bootstrap_allocations)
    num_projects = data_matrix.shape[1]

    quantiles = np.quantile(data_matrix, [0.05, .25, .50, .75, 0.95], axis=0)

    blues = mpl.colormaps['Blues']
    plt.figure()
    plt.fill_between(np.arange(num_projects), quantiles[0], quantiles[4], color=blues(0.2), label='90% CI')
    plt.fill_between(np.arange(num_projects), quantiles[0], quantiles[2], color=blues(0.5), label='IQR')
    plt.plot(quantiles[1], label='Median', color=blues(0.8))
    # overlay actual distribution
    data=np.sort(np.asarray(list(actual_allocation.values())))[::-1]
    plt.plot(data, label='Actual', color='k')

    plt.legend()
    plt.xlabel('Project Index')
    plt.ylabel('FIL')
    plt.title('Bootstrapped Allocation Distribution')
    if save_fp is not None:
        plt.savefig(save_fp)

def plot_allocations_sunburst(df, figsize=(800,800), show_project_labels=True, save_fp=None):
    """
    Create an interactive sunburst plot of project allocations using Plotly.
    """
    # Add debug prints to check input data
    print("Input DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("Sample of input data:")
    print(df.head())
    
    # Verify required columns exist
    required_cols = ['Project Name', 'Final Allocation (FIL)', 'category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        print("Warning: Found null values:")
        print(null_counts)
        # Remove rows with null values
        df = df.dropna(subset=required_cols)
        print(f"Shape after removing nulls: {df.shape}")
    
    # Sort data by allocation amount within each category
    df_sorted = df.sort_values(['category', 'Final Allocation (FIL)'], ascending=[True, False])
    
    # Print debug info about categories and allocations
    print("\nCategory summary:")
    print(df_sorted.groupby('category')['Final Allocation (FIL)'].agg(['count', 'sum']))
    
    def obscure_name(name, index, length=15):
        """Create a unique obscured version of the project name"""
        special_chars = '~!@#$%^&*()+-/\\,.\"'
        rng = random.Random(index)
        result = [rng.choice(special_chars) for _ in range(length)]
        idx_str = str(index).zfill(2)
        mid_point = length // 2
        result[mid_point:mid_point+2] = idx_str
        if len(name) > 4:
            result[4] = name[len(name)//3]
            result[11] = name[2*len(name)//3]
        return ''.join(result)
    
    # Create lists for the sunburst plot
    categories = df_sorted['category'].unique().tolist()
    if show_project_labels:
        project_labels = df_sorted['Project Name'].tolist()
    else:
        project_labels = [obscure_name(name, i) for i, name in enumerate(df_sorted['Project Name'])]
    
    labels = categories + project_labels
    parents = [''] * len(categories) + df_sorted['category'].tolist()
    
    # Create values list
    category_sums = df_sorted.groupby('category')['Final Allocation (FIL)'].sum().tolist()
    project_values = df_sorted['Final Allocation (FIL)'].tolist()
    values = category_sums + project_values
    
    # Print debug info about plot data
    print("\nPlot data summary:")
    print(f"Number of categories: {len(categories)}")
    print(f"Number of projects: {len(project_labels)}")
    print(f"Total length of labels: {len(labels)}")
    print(f"Total length of parents: {len(parents)}")
    print(f"Total length of values: {len(values)}")
    
    # Format hover text
    total_funding = df_sorted['Final Allocation (FIL)'].sum()
    hover_text = []
    
    # Hover text for categories
    for cat, val in zip(categories, category_sums):
        hover_text.append(
            f'Category: {cat}<br>'
            f'Amount: {val:,.2f} FIL<br>'
            f'Percentage: {(val/total_funding)*100:.1f}%'
        )
    
    # Hover text for projects
    for proj, val, cat in zip(df_sorted['Project Name'], project_values, df_sorted['category']):
        hover_text.append(
            f'Project: {proj}<br>'
            f'Category: {cat}<br>'
            f'Amount: {val:,.2f} FIL<br>'
            f'Percentage: {(val/total_funding)*100:.1f}%'
        )
    
    # Get colors
    n_colors = len(df_sorted) + len(categories)
    colors = sns.color_palette("RdYlBu_r", n_colors=n_colors)
    colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
              for r, g, b in colors]
    
    # Create the sunburst plot
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        hovertext=hover_text,
        hoverinfo='text',
        branchvalues='total',
        marker=dict(colors=colors),
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'FIL-RetroPGF-3 Project Allocations (FIL)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        width=figsize[0],
        height=figsize[1],
        margin=dict(t=70, l=0, r=0, b=0)
    )

    if save_fp is not None:
        fig.write_image(save_fp)
    
    return fig

def plot_category_sunburst(df, figsize=(800,800), save_fp=None):
    """
    Create a sunburst plot showing only categories with their funding percentages.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'Final Allocation (FIL)' and 'category' columns
    figsize (tuple): Figure dimensions (width, height)
    save_fp (str): Optional filepath to save the plot
    """
    # Calculate category totals and percentages
    category_totals = df.groupby('category')['Final Allocation (FIL)'].sum().sort_values(ascending=False)
    total_funding = category_totals.sum()
    percentages = (category_totals / total_funding * 100).round(1)
    
    # Create labels with percentages
    labels = [f"{cat}\n({pct}%)" for cat, pct in percentages.items()]
    
    # Create the basic sunburst structure
    parents = [''] * len(labels)  # All categories at root
    values = category_totals.values.tolist()
    
    # Format hover text
    hover_text = [
        f'Category: {cat}<br>'
        f'Amount: {val:,.2f} FIL<br>'
        f'Percentage: {pct}%'
        for cat, val, pct in zip(category_totals.index, values, percentages)
    ]
    
    # Get colors - using fewer colors since we only have categories
    colors = sns.color_palette("RdYlBu_r", n_colors=len(labels))
    colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
              for r, g, b in colors]
    
    # Create the sunburst plot
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        hovertext=hover_text,
        hoverinfo='text',
        branchvalues='total',
        marker=dict(colors=colors),
        textfont=dict(size=14),  # Increase text size for better readability
        insidetextorientation='horizontal'  # Make text horizontal for better readability
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Funding Distribution by Category',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        width=figsize[0],
        height=figsize[1],
        margin=dict(t=50, l=0, r=0, b=0)
    )

    if save_fp is not None:
        fig.write_image(save_fp)
    
    return fig

def plot_category_metrics(df, figsize=(15, 5), style='bar', save_fp=None):
    """
    Plot category-level metrics: total votes, total funding, and average funding per project.
    
    Args:
        df: DataFrame with 'category', 'Final Allocation (FIL)', and vote count information
        figsize: Tuple for figure size
        style: 'bar' for bar charts or 'dot' for connected dot plot
        save_fp: Optional filepath to save the plot
    """
    # Calculate metrics per category
    category_metrics = df.groupby('category').agg({
        'Vote Count': 'sum',
        'Final Allocation (FIL)': ['sum', 'mean']
    }).round(2)
    
    # Rename columns for clarity
    category_metrics.columns = ['Total Votes', 'Total Funding', 'Avg Funding per Project']
    
    # Calculate overall average funding per project
    overall_avg_funding = df['Final Allocation (FIL)'].mean()
    
    # Sort by total funding
    category_metrics = category_metrics.sort_values('Total Funding', ascending=True)
    
    if style == 'bar':
        # Create three subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Define specific colors
        plot_colors = ['#E41A1C',    # Strong red
                      '#377EB8',      # Strong blue
                      '#4DAF4A']      # Strong green
        
        # Plot 1: Total Votes
        category_metrics['Total Votes'].plot(kind='barh', ax=axes[0], color=plot_colors[0])
        axes[0].set_title('Total Votes by Category')
        axes[0].set_xlabel('Number of Votes')
        axes[0].set_ylabel('')
        # Plot 2: Total Funding
        (category_metrics['Total Funding']/1e3).plot(kind='barh', ax=axes[1], color=plot_colors[1])
        axes[1].set_title('Total Funding by Category')
        axes[1].set_xlabel('FIL (thousands)')
        axes[1].set_ylabel('')
        
        # Plot 3: Average Funding with reference line
        category_metrics['Avg Funding per Project'].plot(kind='barh', ax=axes[2], color=plot_colors[2])
        axes[2].set_title('Avg Funding per Project')
        axes[2].set_xlabel('FIL')
        axes[2].set_ylabel('')
        # Add reference line for overall average
        axes[2].axvline(x=overall_avg_funding, color='red', linestyle='--', alpha=0.7)
        
        # Position the text in the middle of the y-axis
        y_mid = len(category_metrics) / 2
        axes[2].text(overall_avg_funding * 1.1, y_mid, 
                    f'Overall Avg:\n{overall_avg_funding:,.0f} FIL', 
                    verticalalignment='center', 
                    horizontalalignment='left',
                    color='red')
        
        # Adjust layout
        plt.tight_layout()
        
    else:  # dot plot implementation remains the same
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize each metric to [0,1] for comparison
        normalized_metrics = category_metrics.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        # Plot lines and points using the same explicit colors
        markers = ['o', 's', 'D']
        
        for (metric, values), color, marker in zip(normalized_metrics.items(), plot_colors, markers):
            ax.plot(values.index, values.values, 'o-', 
                   label=metric, color=color, marker=marker, markersize=10)
            
            # Add value labels
            for i, v in enumerate(category_metrics[metric]):
                if metric == 'Total Funding':
                    label = f'{v/1000:.0f}k'
                elif metric == 'Avg Funding per Project':
                    label = f'{v/1000:.1f}k'
                else:
                    label = f'{v:.0f}'
                ax.text(i, normalized_metrics[metric][i], label, 
                       ha='left', va='bottom')
        
        # Customize plot
        ax.set_title('Category Metrics Comparison (Normalized Scale)', pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels for better readability
        plt.xticks(range(len(category_metrics)), category_metrics.index, rotation=45, ha='right')
        
        # Remove y-axis as it shows normalized values
        ax.set_ylabel('Normalized Scale')
        
        # Adjust layout
        plt.tight_layout()
    
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight', dpi=300)
    
    return fig

def plot_quorum_sweep(df, quorum_range=range(1, 11), actual_quorum=6, min_funding=250, total_funding=270000, figsize=(12, 5), save_fp=None):
    """
    Plot how allocation distribution changes with different quorum values.
    
    Args:
        df: DataFrame with voting data
        quorum_range: Range of quorum values to test
        actual_quorum: The quorum value actually used (will be highlighted)
        figsize: Figure size tuple
        save_fp: Optional filepath to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Store results for each quorum
    distributions = []
    funded_projects = []
    
    # Create color map using RdPu (red to purple, less bright)
    cmap = plt.cm.RdPu
    norm = plt.Normalize(min(quorum_range), max(quorum_range))
    
    for quorum in quorum_range:
        allocations, _, _, _, _, _ = run_allocation_analysis(
            df,
            quorum_cutoff=quorum,
            min_funding=min_funding,
            total_funding=total_funding,
            verbose=False
        )
        
        if allocations:
            data = np.sort(np.asarray(list(allocations.values())))[::-1]
            distributions.append(data)
            funded_projects.append(len(data))
            
            # Set line properties based on whether this is the actual quorum
            if quorum == actual_quorum:
                color = 'black'  # Changed to black for better contrast
                alpha = 1.0
                zorder = 10
                linewidth = 2.5
            else:
                color = cmap(norm(quorum))
                alpha = 0.7
                zorder = 5
                linewidth = 1.5
            
            # Plot distribution
            ax1.plot(np.arange(len(data)), data, 
                    color=color,
                    alpha=alpha,
                    zorder=zorder,
                    linewidth=linewidth)
    
    ax1.set_xlabel('Project Index')
    ax1.set_ylabel('Allocation (FIL)')
    ax1.set_title('Allocation Distributions by Quorum')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Quorum Value')
    
    # Add legend only for actual quorum
    ax1.plot([], [], color='black', linewidth=2.5, 
            label=f'Actual Quorum ({actual_quorum})')
    ax1.legend(loc='upper right')
    
    # Plot number of funded projects vs quorum
    ax2.plot(quorum_range, funded_projects, 'o-', color='#1f77b4')
    # Highlight the actual quorum point
    ax2.plot(actual_quorum, funded_projects[actual_quorum-quorum_range[0]], 
            'o', color='black', markersize=10, label=f'Actual Quorum ({actual_quorum})')
    
    ax2.set_xlabel('Quorum Requirement')
    ax2.set_ylabel('Number of Funded Projects')
    ax2.set_title('Funded Projects vs Quorum')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Ensure integer ticks for quorum axis
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    if save_fp:
        plt.savefig(save_fp, bbox_inches='tight')
    return fig

def plot_scoring_comparison(df, quorum_cutoff=6, min_funding=250, total_funding=270000, figsize=(10, 5), save_fp=None):
    """
    Compare different scoring functions (sum, mean, median) for allocations.
    
    Args:
        df: DataFrame with voting data
        figsize: Figure size tuple
        save_fp: Optional filepath to save plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get allocations for each scoring method
    scoring_methods = ['sum', 'mean', 'median']
    colors = ['black', '#1f77b4', '#2ca02c']  # black, blue, green
    
    for method, color in zip(scoring_methods, colors):
        allocations, eliminated, _, _ = get_allocations(
            df,
            quorum_cutoff=quorum_cutoff,
            min_funding=min_funding,
            total_funding=total_funding,
            scoring_fn=method
        )
        
        data = np.sort(np.asarray(list(allocations.values())))[::-1]
        
        # Plot with different styles for each method
        linewidth = 2.5 if method == 'sum' else 1.5  # Make sum (original) stand out
        alpha = 1.0 if method == 'sum' else 0.8
        
        ax.plot(np.arange(len(data)), data, 
               label=f'{method.capitalize()} ({len(data)} projects)',
               color=color,
               linewidth=linewidth,
               alpha=alpha)
    
    ax.set_xlabel('Project Index')
    ax.set_ylabel('Allocation (FIL)')
    ax.set_title('Allocation Distributions by Scoring Method')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_fp:
        plt.savefig(save_fp, bbox_inches='tight')
    return fig

def plot_temperature_analysis(df, quorum_cutoff=6, min_funding=250, total_funding=270000, hot_cold_pct=0.1, figsize=(12, 5), save_fp=None):
    """
    Analyze how removing hot/cold voters affects allocation distribution.
    
    Args:
        df: DataFrame with voting data
        hot_cold_pct: Percentage of hot/cold voters to remove (e.g., 0.1 for top/bottom 10%)
        figsize: Figure size tuple
        save_fp: Optional filepath to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate voter temperatures (number of projects voted on)
    voter_temps = df.groupby('voterId').size().sort_values()
    num_voters_to_remove = int(len(voter_temps) * hot_cold_pct)
    
    # Get hot and cold voter IDs
    hot_voters = voter_temps.head(num_voters_to_remove).index  # Bottom 10% (voted on fewest projects)
    cold_voters = voter_temps.tail(num_voters_to_remove).index  # Top 10% (voted on most projects)
    
    # Original distribution
    allocations_orig, _, _, _, _, _ = run_allocation_analysis(
        df,
        quorum_cutoff=quorum_cutoff,
        min_funding=min_funding,
        total_funding=total_funding,
        verbose=False
    )
    data_orig = np.sort(np.asarray(list(allocations_orig.values())))[::-1]
    
    # Distribution without cold voters
    df_no_cold = df[~df['voterId'].isin(cold_voters)]
    allocations_no_cold, _, _, _, _, _ = run_allocation_analysis(
        df_no_cold,
        quorum_cutoff=quorum_cutoff,
        min_funding=min_funding,
        total_funding=total_funding,
        verbose=False
    )
    data_no_cold = np.sort(np.asarray(list(allocations_no_cold.values())))[::-1]
    
    # Distribution without hot voters
    df_no_hot = df[~df['voterId'].isin(hot_voters)]
    allocations_no_hot, _, _, _, _, _ = run_allocation_analysis(
        df_no_hot,
        quorum_cutoff=quorum_cutoff,
        min_funding=min_funding,
        total_funding=total_funding,
        verbose=False
    )
    data_no_hot = np.sort(np.asarray(list(allocations_no_hot.values())))[::-1]
    
    # Distribution without both
    df_no_both = df[~df['voterId'].isin(hot_voters) & ~df['voterId'].isin(cold_voters)]
    allocations_no_both, _, _, _, _, _ = run_allocation_analysis(
        df_no_both,
        quorum_cutoff=quorum_cutoff,
        min_funding=min_funding,
        total_funding=total_funding,
        verbose=False
    )
    data_no_both = np.sort(np.asarray(list(allocations_no_both.values())))[::-1]
    
    # Plot distributions with clearer labels and original in bold black
    ax1.plot(np.arange(len(data_orig)), data_orig, 
             label='Original Distribution', 
             color='black', 
             linewidth=2.5,
             zorder=5)
    
    ax1.plot(np.arange(len(data_no_cold)), data_no_cold, 
             label=f'Removed Top {hot_cold_pct*100:.0f}% Most Active Voters', 
             linewidth=1.5,
             alpha=0.8)
    
    ax1.plot(np.arange(len(data_no_hot)), data_no_hot, 
             label=f'Removed Bottom {hot_cold_pct*100:.0f}% Least Active Voters', 
             linewidth=1.5,
             alpha=0.8)
    
    ax1.plot(np.arange(len(data_no_both)), data_no_both, 
             label=f'Removed Both Top & Bottom {hot_cold_pct*100:.0f}%', 
             linewidth=1.5,
             alpha=0.8)
    
    ax1.set_xlabel('Project Index')
    ax1.set_ylabel('Allocation (FIL)')
    ax1.set_title('Allocation Distributions by Voter Activity Level')
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize='small', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot voter temperature distribution
    sns.histplot(voter_temps, ax=ax2)
    hot_threshold = voter_temps.quantile(hot_cold_pct)
    cold_threshold = voter_temps.quantile(1-hot_cold_pct)
    
    ax2.axvline(hot_threshold, color='r', linestyle='--', 
                label=f'Bottom {hot_cold_pct*100:.0f}% Threshold\n({hot_threshold:.0f} projects)')
    ax2.axvline(cold_threshold, color='b', linestyle='--', 
                label=f'Top {hot_cold_pct*100:.0f}% Threshold\n({cold_threshold:.0f} projects)')
    
    ax2.set_xlabel('Number of Projects Voted On')
    ax2.set_ylabel('Number of Voters')
    ax2.set_title('Distribution of Voter Activity Levels')
    ax2.legend()
    
    plt.tight_layout()
    if save_fp:
        plt.savefig(save_fp, bbox_inches='tight')
    return fig