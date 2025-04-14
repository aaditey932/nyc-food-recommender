import pandas as pd
from typing import Dict, Optional

def get_highest_rated_restaurant(metadata: Dict, max_time: float) -> Optional[Dict]:
    """
    Get the highest-rated restaurant from metadata with order time under max_time.
    
    Args:
        metadata: Restaurant metadata dictionary (from create_restaurant_metadata())
                  Expected structure is a dict of columns where each key is a column name
                  and each value is a dict with restaurant names as keys.
        max_time: Max total order time in minutes
    
    Returns:
        Dictionary with the restaurant name and other metadata, or None if no match.
    """
    # Convert the dictionary of columns back to a DataFrame.
    df = pd.DataFrame.from_dict(metadata, orient='columns')
    
    # At this point, the DataFrame's index is the restaurant names.
    # Reset the index to turn it into a column named "restaurant_name".
    df = df.reset_index().rename(columns={'index': 'restaurant_name'})
    
    # Filter restaurants by the provided maximum order time.
    df_filtered = df[df['average_order_time'] <= max_time]
    if df_filtered.empty:
        return None

    # Sort by average rating in descending order and pick the top restaurant.
    top_restaurant = df_filtered.sort_values(by='average_rating', ascending=False).iloc[0]
    return top_restaurant.to_dict()