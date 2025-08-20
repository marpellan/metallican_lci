from typing import List, Tuple
import pandas as pd


def merge_without_suffixes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Tuple[str, str] = ("main_id", "facility_group_id"),
    how: str = "left",
    right_prefix: str = ""  # leave empty to add only non-overlapping cols
) -> pd.DataFrame:
    """
    Merge two DataFrames on keys, adding only *new* columns from right to avoid _x/_y.
    - Ensures right has unique keys to prevent row multiplication.
    - Drops overlapping non-key columns from right before merge.
    - Returns a clean merged DataFrame with no suffixes.
    """
    key_cols = list(keys)

    # Ensure key types align
    for k in key_cols:
        if k in left.columns and k in right.columns:
            # Cast to string to be robust to mixed types; change to category/int if you prefer
            left[k] = left[k].astype(str)
            right[k] = right[k].astype(str)

    # Deduplicate right on keys (keep first occurrence)
    if not right.duplicated(subset=key_cols, keep=False).any():
        right_dedup = right.copy()
    else:
        # If duplicates exist, keep the first row per keys (you can change strategy if needed)
        right_dedup = right.drop_duplicates(subset=key_cols, keep="first")

    # Determine overlapping non-key columns and drop them from right
    overlap = [c for c in right_dedup.columns if c in left.columns and c not in key_cols]
    right_cols_to_use = [c for c in right_dedup.columns if c not in overlap]

    # Optionally add a prefix to new columns from right (disabled by default to keep names clean)
    if right_prefix:
        rename_map = {
            c: f"{right_prefix}{c}" for c in right_cols_to_use if c not in key_cols
        }
        right_dedup = right_dedup[right_cols_to_use].rename(columns=rename_map)
    else:
        right_dedup = right_dedup[right_cols_to_use]

    # Perform merge
    merged = left.merge(right_dedup, on=key_cols, how=how, validate=None)

    return merged


def normalize_by_production(df, production_df, value_col='value', prod_col='value_tonnes', prod_agg='sum'):
    df = df.copy()
    # ensure numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    prod = production_df.copy()
    prod[prod_col] = pd.to_numeric(prod[prod_col], errors='coerce')

    # aggregate to unique per key
    main_prod = (prod.dropna(subset=['main_id'])
                    .groupby('main_id', as_index=False)[prod_col]
                    .agg(prod_agg)
                    .rename(columns={prod_col: 'value_tonnes_main'}))
    group_prod = (prod.dropna(subset=['facility_group_id'])
                     .groupby('facility_group_id', as_index=False)[prod_col]
                     .agg(prod_agg)
                     .rename(columns={prod_col: 'value_tonnes_group'}))

    # safe 1:1 merges
    out = df.merge(main_prod, on='main_id', how='left').merge(group_prod, on='facility_group_id', how='left')

    # prefer main_id match, fallback to facility_group_id
    out['value_tonnes_match'] = out['value_tonnes_main'].combine_first(out['value_tonnes_group'])
    out['value_normalized'] = out[value_col] / out['value_tonnes_match']

    # diagnostics
    out['normalization_key'] = None
    out.loc[out['value_tonnes_main'].notna(), 'normalization_key'] = 'main_id'
    out.loc[out['value_tonnes_main'].isna() & out['value_tonnes_group'].notna(), 'normalization_key'] = 'facility_group_id'
    return out


def get_info_for_ids(df, id_pairs):
    # Extract facility_group_ids from id_pairs
    facility_group_ids = {id_pair[1] for id_pair in id_pairs if pd.notna(id_pair[1])}

    # Create a boolean mask for rows that match any of the id pairs or have a facility_group_id in the set
    mask = df.apply(lambda row: (
        (row['main_id'], row['facility_group_id']) in id_pairs or
        row['facility_group_id'] in facility_group_ids
    ), axis=1)

    # Use the mask to filter the DataFrame
    filtered_df = df[mask]
    return filtered_df


import pandas as pd
import plotly.graph_objects as go
import io

def create_sankey_diagram(data):
    """
    Generates a Sankey diagram from the provided DataFrame.

    Args:
        data (pd.DataFrame): A DataFrame containing the raw data.

    Returns:
        plotly.graph_objects.Figure: The generated Sankey figure.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # Data cleaning and preparation
    # 1. Clean the 'commodities' column to handle different orderings of the same items.
    #    This ensures "Gold, copper, silver" and "Copper, silver, gold" are treated the same.
    df['commodities_cleaned'] = df['commodities'].apply(
        lambda x: ', '.join(sorted([c.strip().lower() for c in str(x).split(',')])) if pd.notna(x) else 'Unknown Commodities'
    )

    # 2. Define the columns to be used for the Sankey flow
    columns = [
        'commodities_cleaned',
        'province',
        'mining_processing_type',
        'mining_method',
        'mining_submethod'
    ]

    # Replace NaN values in the flow columns with a specific 'Unknown' label for each column
    for col in columns:
        # We handle commodities_cleaned separately, so we skip it here.
        if col != 'commodities_cleaned':
            df[col] = df[col].fillna(f'Unknown {col.replace("_", " ").title()}')

    # 3. Apply custom prefixes to ensure uniqueness and readability
    df['mining_processing_type'] = df['mining_processing_type'].apply(lambda x: f"NRC: {x}")
    df['mining_method'] = df['mining_method'].apply(lambda x: f"MDO: {x}")
    df['mining_submethod'] = df['mining_submethod'].apply(lambda x: f"MDO_s: {x}")

    # Create a list of all unique labels for the nodes
    labels = []
    for col in columns:
        labels.extend(df[col].unique().tolist())

    labels = sorted(list(set(labels)))

    # Create a mapping from label to an index for Plotly
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Initialize lists to store the source, target, and value of each link
    source = []
    target = []
    value = []

    # Iterate through the defined flow path and create the links
    for i in range(len(columns) - 1):
        col1 = columns[i]
        col2 = columns[i+1]

        # Group by the source and target nodes to count the links
        link_counts = df.groupby([col1, col2]).size().reset_index(name='count')

        for _, row in link_counts.iterrows():
            source_label = row[col1]
            target_label = row[col2]
            link_value = row['count']

            # Ensure both source and target labels exist in our map
            if pd.notna(source_label) and pd.notna(target_label):
                source.append(label_to_index[source_label])
                target.append(label_to_index[target_label])
                value.append(link_value)

    # Create the Sankey diagram using plotly
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,  # Increased pad for vertical layout
            thickness=25, # Increased thickness for vertical layout
            line=dict(color="black", width=0.5),
            label=labels,
            hovertemplate='%{label}<br>Value: %{value} facilities<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate='From %{source.label}<br>To %{target.label}<br>Count: %{value}<extra></extra>'
        ),
        orientation='v' # Set the orientation to vertical here
    )])

    # Update layout for a better visualization
    fig.update_layout(
        title_text="Sankey Diagram of Mining Facilities",
        font=dict(size=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

