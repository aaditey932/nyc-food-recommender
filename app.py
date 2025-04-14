import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional

# Set page title and configuration
st.set_page_config(
    page_title="NYC Food Recommender",
    page_icon="🍔",
    layout="wide"
)

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the food order data.
    
    Args:
        file_path: Path to the CSV file containing food order data
    
    Returns:
        Preprocessed DataFrame with cleaned ratings and added total_order_time
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Add total order time
    df['total_order_time'] = df['food_preparation_time'] + df['delivery_time']
    
    # Clean rating column - replace 'Not given' with None and convert to float
    df['rating'] = df['rating'].replace('Not given', None)
    df['rating'] = df['rating'].astype('float64')
    
    # Fill missing ratings with the restaurant's average rating
    df['rating'] = df.groupby(['restaurant_name'], sort=False)['rating'].apply(
        lambda x: x.fillna(x.mean())
    ).values
    
    # Drop remaining NaN values
    df.dropna(inplace=True)
    
    return df

def filter_popular_restaurants(df: pd.DataFrame, min_ratings: int = 40) -> pd.DataFrame:
    """
    Filter restaurants based on popularity (number of ratings).
    
    Args:
        df: Input DataFrame with restaurant data
        min_ratings: Minimum number of ratings required for a restaurant
    
    Returns:
        DataFrame containing only popular restaurants
    """
    restaurant_counts = df.groupby('restaurant_name')['rating'].size().reset_index()
    popular_restaurants = restaurant_counts[restaurant_counts['rating'] > min_ratings]['restaurant_name'].values
    return df[df['restaurant_name'].isin(popular_restaurants)]

def create_restaurant_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table with restaurants as rows and customers as columns.
    
    Args:
        df: Filtered DataFrame with restaurant and customer data
    
    Returns:
        Pivot table for restaurant-based recommendations
    """
    pivot_table = df.pivot_table(index='restaurant_name', columns='customer_id', values='rating')
    pivot_table.fillna(0, inplace=True)
    return pivot_table

def create_restaurant_metadata(df: pd.DataFrame) -> Dict:
    """
    Create a dictionary with restaurant metadata for display purposes.
    
    Args:
        df: Filtered DataFrame with restaurant data
    
    Returns:
        Dictionary containing restaurant metadata
    """
    restaurant_data = df.groupby(['restaurant_name', 'cuisine_type']).aggregate({
        'rating': 'mean', 
        'cost_of_the_order': 'mean', 
        'order_id': 'size',
        'total_order_time': 'mean'
    }).reset_index(level='cuisine_type')
    
    restaurant_data = restaurant_data.rename(columns={
        'rating': 'average_rating', 
        'cost_of_the_order': 'average_order_cost', 
        'order_id': 'rating_count',
        'total_order_time': 'average_order_time'
    })
    
    # Round numerical columns for better display
    for col in ['average_rating', 'average_order_cost', 'average_order_time']:
        restaurant_data[col] = restaurant_data[col].round(2)
    
    return restaurant_data.to_dict()

def recommend_similar_restaurants(
    restaurant_df: pd.DataFrame, 
    restaurant_name: str, 
    restaurant_limit: int = 10
) -> List[Tuple[int, float]]:
    """
    Generate restaurant recommendations based on similarity.
    
    Args:
        restaurant_df: Pivot table with restaurant and customer data
        restaurant_name: Name of the restaurant to base recommendations on
        restaurant_limit: Number of recommendations to generate
    
    Returns:
        List of tuples containing restaurant indices and similarity scores
    """
    # Find the index of the input restaurant
    restaurant_index = np.where(restaurant_df.index == restaurant_name)[0][0]
    
    # Calculate cosine similarity between all restaurants
    similarities = cosine_similarity(restaurant_df)
    
    # Get the most similar restaurants (excluding the input restaurant itself)
    recommended_restaurants = sorted(
        list(enumerate(similarities[restaurant_index])), 
        key=lambda x: x[1], 
        reverse=True
    )[1:restaurant_limit+1]
    
    return recommended_restaurants

def get_customer_favorite_restaurant(df: pd.DataFrame, customer_id: str) -> str:
    """
    Find a customer's favorite restaurant based on highest rating.
    
    Args:
        df: DataFrame with customer ratings
        customer_id: ID of the customer
    
    Returns:
        Name of the customer's favorite restaurant
    """
    customer_data = df[df['customer_id'] == customer_id]
    if customer_data.empty:
        return None
    return customer_data.sort_values(by='rating', ascending=False)['restaurant_name'].values[0]

def filter_recommendations_by_time(
    recommendations: List[Dict], 
    max_time: float
) -> List[Dict]:
    """
    Filter restaurant recommendations based on total order time.
    
    Args:
        recommendations: List of restaurant recommendation dictionaries
        max_time: Maximum acceptable order time in minutes
    
    Returns:
        Filtered list of recommendations
    """
    return [r for r in recommendations if r.get('average_order_time', 0) <= max_time]

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
    # Specify orient='columns' for clarity (the default works too).
    df = pd.DataFrame.from_dict(metadata, orient='columns')
    
    # At this point, the DataFrame's index is the restaurant names.
    # Reset the index to turn it into a column named "index", then rename it.
    df = df.reset_index().rename(columns={'index': 'restaurant_name'})
    
    # Filter restaurants by the provided maximum order time.
    df_filtered = df[df['average_order_time'] <= max_time]
    if df_filtered.empty:
        return None

    # Sort by average rating in descending order and pick the top restaurant.
    top_restaurant = df_filtered.sort_values(by='average_rating', ascending=False).iloc[0]
    return top_restaurant.to_dict()

def main():
    # App title
    st.title("NYC Food Recommender System 🍔")
    st.markdown("Get personalized restaurant recommendations based on your preferences")
    
    # File upload option or use default path
    uploaded_file = st.file_uploader("Upload your food order CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data from uploaded file
        df = pd.read_csv(uploaded_file)
    else:
        # Use default path (for development)
        st.info("No file uploaded. Using default dataset.")
        try:
            # Try to load from a likely default path
            default_path = "data/food_order.csv"
            if os.path.exists(default_path):
                df = pd.read_csv(default_path)
            else:
                st.error("Default dataset not found. Please upload a CSV file.")
                return
        except Exception as e:
            st.error(f"Error loading default dataset: {e}")
            return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
    "Restaurant-Based Recommendations", 
    "Customer-Based Recommendations",
    "Best Rated Restaurant"
])
    
    # Process data
    with st.spinner("Processing data..."):
        # Preprocess the data
        df = load_and_preprocess_data("data/food_order.csv" if uploaded_file is None else uploaded_file)
        
        # Filter popular restaurants
        filtered_df = filter_popular_restaurants(df)
        
        # Create pivot table for recommendations
        rest_df = create_restaurant_pivot_table(filtered_df)
        
        # Create restaurant metadata
        restaurant_data = create_restaurant_metadata(filtered_df)
        
        # Get unique restaurants and customers for selection
        restaurant_list = rest_df.index.tolist()
        customer_list = filtered_df['customer_id'].unique().tolist()
    
    # Restaurant-Based Recommendations Tab
    with tab1:
        st.header("Find Similar Restaurants")
        
        # Restaurant selection
        selected_restaurant = st.selectbox(
            "Select your favorite restaurant:", 
            restaurant_list,
            index=0
        )
        
        # Number of recommendations slider
        num_recommendations = st.slider(
            "Number of recommendations:", 
            min_value=1, 
            max_value=20, 
            value=10
        )
        
        # Order time filter
        max_order_time = st.slider(
            "Maximum total order time (minutes):", 
            min_value=10, 
            max_value=90, 
            value=60,
            step=5
        )
        
        if st.button("Get Restaurant Recommendations"):
            # Get recommendations
            recommended_restaurants = recommend_similar_restaurants(
                rest_df, 
                selected_restaurant, 
                num_recommendations
            )
            
            # Display favorite restaurant info
            st.subheader("Your Selected Restaurant:")
            st.write(f"**{selected_restaurant}**")
            st.write(f"Cuisine type: {restaurant_data['cuisine_type'][selected_restaurant]}")
            st.write(f"Average rating: {restaurant_data['average_rating'][selected_restaurant]}")
            st.write(f"Average order cost: ${restaurant_data['average_order_cost'][selected_restaurant]}")
            st.write(f"Average order time: {restaurant_data['average_order_time'][selected_restaurant]} minutes")
            
            # Process recommendations
            recs = []
            for idx, similarity in recommended_restaurants:
                rest_name = rest_df.index[idx]
                rec = {
                    "name": rest_name,
                    "cuisine": restaurant_data['cuisine_type'][rest_name],
                    "rating": restaurant_data['average_rating'][rest_name],
                    "cost": restaurant_data['average_order_cost'][rest_name],
                    "average_order_time": restaurant_data['average_order_time'][rest_name],
                    "similarity": round(similarity * 100, 2)
                }
                recs.append(rec)
            
            # Filter by order time
            filtered_recs = [r for r in recs if r['average_order_time'] <= max_order_time]
            
            # Display recommendations
            st.subheader(f"Top {len(filtered_recs)} Recommended Restaurants:")
            
            if not filtered_recs:
                st.warning("No restaurants match your order time criteria. Try increasing the maximum order time.")
            else:
                # Create a table of recommendations
                rec_df = pd.DataFrame(filtered_recs)
                rec_df.columns = ['Restaurant Name', 'Cuisine Type', 'Avg. Rating', 'Avg. Cost ($)', 'Avg. Order Time (min)', 'Similarity (%)']
                st.dataframe(rec_df.sort_values('Similarity (%)', ascending=False), use_container_width=True)
    
    # Customer-Based Recommendations Tab
    with tab2:
        st.header("Personalized Customer Recommendations")
        
        # Customer ID selection
        selected_customer = st.selectbox(
            "Select a customer ID:", 
            [""] + customer_list,
            index=0
        )
        
        # Custom customer ID input
        custom_customer = st.text_input("Or type a customer ID:")
        
        # Use custom input if provided
        if custom_customer:
            customer_to_use = custom_customer
        else:
            customer_to_use = selected_customer
        
        # Number of recommendations slider
        num_customer_recs = st.slider(
            "Number of recommendations:", 
            min_value=1, 
            max_value=20, 
            value=10,
            key="customer_recs_slider"
        )
        
        # Order time filter
        max_customer_order_time = st.slider(
            "Maximum total order time (minutes):", 
            min_value=10, 
            max_value=90, 
            value=60,
            step=5,
            key="customer_time_slider"
        )
        
        if st.button("Get Customer Recommendations") and customer_to_use:
            # Find the customer's favorite restaurant
            favorite_restaurant = get_customer_favorite_restaurant(filtered_df, customer_to_use)
            
            if favorite_restaurant is None:
                st.error(f"Customer ID '{customer_to_use}' not found in the dataset.")
            else:
                # Get recommendations based on favorite restaurant
                recommended_restaurants = recommend_similar_restaurants(
                    rest_df, 
                    favorite_restaurant, 
                    num_customer_recs
                )
                
                # Display customer's favorite restaurant
                st.subheader(f"Customer's Favorite Restaurant:")
                st.write(f"**{favorite_restaurant}**")
                st.write(f"Cuisine type: {restaurant_data['cuisine_type'][favorite_restaurant]}")
                st.write(f"Average rating: {restaurant_data['average_rating'][favorite_restaurant]}")
                st.write(f"Average order cost: ${restaurant_data['average_order_cost'][favorite_restaurant]}")
                st.write(f"Average order time: {restaurant_data['average_order_time'][favorite_restaurant]} minutes")
                
                # Process recommendations
                customer_recs = []
                for idx, similarity in recommended_restaurants:
                    rest_name = rest_df.index[idx]
                    rec = {
                        "name": rest_name,
                        "cuisine": restaurant_data['cuisine_type'][rest_name],
                        "rating": restaurant_data['average_rating'][rest_name],
                        "cost": restaurant_data['average_order_cost'][rest_name],
                        "average_order_time": restaurant_data['average_order_time'][rest_name],
                        "similarity": round(similarity * 100, 2)
                    }
                    customer_recs.append(rec)
                
                # Filter by order time
                filtered_customer_recs = [r for r in customer_recs if r['average_order_time'] <= max_customer_order_time]
                
                # Display recommendations
                st.subheader(f"Top {len(filtered_customer_recs)} Recommended Restaurants:")
                
                if not filtered_customer_recs:
                    st.warning("No restaurants match your order time criteria. Try increasing the maximum order time.")
                else:
                    # Create a table of recommendations
                    customer_rec_df = pd.DataFrame(filtered_customer_recs)
                    customer_rec_df.columns = ['Restaurant Name', 'Cuisine Type', 'Avg. Rating', 'Avg. Cost ($)', 'Avg. Order Time (min)', 'Similarity (%)']
                    st.dataframe(customer_rec_df.sort_values('Similarity (%)', ascending=False), use_container_width=True)
    # Best Rated Restaurant Tab
    with tab3:
        st.header("Top Rated Restaurant (Simple Recommendation)")
        
        # Time constraint
        best_time_limit = st.slider(
            "Maximum total order time (minutes):", 
            min_value=10, 
            max_value=90, 
            value=60,
            step=5,
            key="best_time_slider"
        )
        
        if st.button("Find Best Rated Restaurant"):
            top_restaurant = get_highest_rated_restaurant(restaurant_data, best_time_limit)
            
            if top_restaurant:
                st.subheader(f"🍽️ {top_restaurant['restaurant_name']}")
                st.write(f"**Cuisine:** {top_restaurant['cuisine_type']}")
                st.write(f"**Average Rating:** {top_restaurant['average_rating']}")
                st.write(f"**Average Cost:** ${top_restaurant['average_order_cost']}")
                st.write(f"**Average Order Time:** {top_restaurant['average_order_time']} minutes")
            else:
                st.warning("No restaurant meets the selected time constraint.")

if __name__ == "__main__":
    main()