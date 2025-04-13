import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import random

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the restaurant order data.
    
    Args:
        file_path (str): Path to the CSV file containing food order data
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Calculate total order time
    df['total_order_time'] = df['food_preparation_time'] + df['delivery_time']
    
    # Handle missing ratings by replacing 'Not given' with None and converting to float
    df['rating'] = df['rating'].replace('Not given', None)
    df['rating'] = df['rating'].astype('float64')
    
    # Fill missing ratings with the mean rating for each restaurant
    df['rating'] = df.groupby(['restaurant_name'], sort=False)['rating'].apply(
        lambda x: x.fillna(x.mean())).values
    
    # Remove any remaining rows with NaN values
    df.dropna(inplace=True)
    
    return df

def filter_popular_restaurants(df, min_ratings=40):
    """
    Filter restaurants based on popularity (number of ratings).
    
    Args:
        df (pd.DataFrame): Input DataFrame with restaurant data
        min_ratings (int): Minimum number of ratings required to include a restaurant
        
    Returns:
        pd.DataFrame: DataFrame with only popular restaurants
    """
    # Count ratings per restaurant
    restaurant_counts = df.groupby('restaurant_name')['rating'].size().sort_values(ascending=False)
    restaurant_counts = restaurant_counts.reset_index()
    
    # Get list of restaurants with ratings above threshold
    restaurant_list = restaurant_counts[restaurant_counts['rating'] > min_ratings]['restaurant_name'].values
    
    # Filter the original DataFrame
    filtered_df = df[df['restaurant_name'].isin(restaurant_list)]
    
    return filtered_df, restaurant_list

def create_restaurant_metadata(filtered_df):
    """
    Create a dictionary with aggregated metadata about each restaurant.
    
    Args:
        filtered_df (pd.DataFrame): DataFrame with filtered restaurant data
        
    Returns:
        dict: Dictionary containing restaurant metadata
    """
    # Group by restaurant and cuisine type to calculate aggregated values
    restaurant_data = filtered_df.groupby(['restaurant_name', 'cuisine_type']).aggregate({
        'rating': 'mean', 
        'cost_of_the_order': 'mean', 
        'order_id': 'size'
    }).reset_index(level='cuisine_type')
    
    # Rename columns for clarity
    restaurant_data = restaurant_data.rename(columns={
        'rating': 'average_rating', 
        'cost_of_the_order': 'average_order_cost', 
        'order_id': 'rating_count'
    })
    
    # Round numerical values for better readability
    restaurant_data[['average_rating', 'average_order_cost']] = restaurant_data[['average_rating', 'average_order_cost']].round(2)
    
    # Convert to dictionary format for easier access
    restaurant_data = restaurant_data.to_dict()
    
    return restaurant_data

def create_restaurant_customer_matrix(filtered_df):
    """
    Create a pivot table of restaurant-customer ratings.
    
    Args:
        filtered_df (pd.DataFrame): DataFrame with filtered restaurant data
        
    Returns:
        pd.DataFrame: Pivot table with restaurants as rows and customers as columns
    """
    rest_df = filtered_df.pivot_table(index='restaurant_name', columns='customer_id', values='rating')
    rest_df.fillna(0, inplace=True)
    return rest_df

def recommend_restaurant(restaurant_df, restaurant_name, restaurant_limit=1):
    """
    Recommend restaurants similar to a given restaurant using cosine similarity.
    
    Args:
        restaurant_df (pd.DataFrame): Pivot table with restaurant-customer ratings
        restaurant_name (str): Name of the restaurant to base recommendations on
        restaurant_limit (int): Number of recommendations to return
        
    Returns:
        list: List of tuples containing (index, similarity_score) for recommended restaurants
    """
    # Find the index of the given restaurant
    restaurant_index = np.where(restaurant_df.index == restaurant_name)[0][0]
    
    # Calculate similarity between all restaurants
    similarities = cosine_similarity(restaurant_df)
    
    # Get the most similar restaurants (excluding the input restaurant itself)
    recommended_restaurants = sorted(
        list(enumerate(similarities[restaurant_index])), 
        key=lambda x: x[1], 
        reverse=True
    )[1:restaurant_limit + 1]
    
    return recommended_restaurants

def display_restaurant_recommendations(rest_df, restaurant_data, restaurant_name, recommendations):
    """
    Display restaurant recommendations with their details.
    
    Args:
        rest_df (pd.DataFrame): Pivot table with restaurant-customer ratings
        restaurant_data (dict): Dictionary with restaurant metadata
        restaurant_name (str): Original restaurant name
        recommendations (list): List of recommended restaurants from recommend_restaurant function
    """
    print(f"Your favorite restaurant: {restaurant_name}, cuisine type: {restaurant_data['cuisine_type'][restaurant_name]}")
    print(f'Customers like {restaurant_name}, also like:')
    for name in recommendations:
        recommended_name = rest_df.index[name[0]]
        print(f"{recommended_name}, "
              f"cuisine type {restaurant_data['cuisine_type'][recommended_name]}, "
              f"average rating {restaurant_data['average_rating'][recommended_name]}, "
              f"average cost {restaurant_data['average_order_cost'][recommended_name]}")

def get_customer_recommendations(filtered_df, rest_df, restaurant_data, customer_id, restaurant_limit=3):
    """
    Get restaurant recommendations for a specific customer.
    
    Args:
        filtered_df (pd.DataFrame): DataFrame with filtered restaurant data
        rest_df (pd.DataFrame): Pivot table with restaurant-customer ratings
        restaurant_data (dict): Dictionary with restaurant metadata
        customer_id (str): ID of the customer to get recommendations for
        restaurant_limit (int): Number of recommendations to return
        
    Returns:
        list: List of recommended restaurant names
    """
    # Find customer's favorite restaurant (highest rated)
    favorite_restaurant = filtered_df[filtered_df['customer_id'] == customer_id].sort_values(
        by='rating', ascending=False)['restaurant_name'].values[0]
    
    # Get recommendations based on the favorite restaurant
    recommendations = recommend_restaurant(rest_df, favorite_restaurant, restaurant_limit)
    
    print(f"{customer_id}'s restaurant, {favorite_restaurant}, they make {restaurant_data['cuisine_type'][favorite_restaurant]} food")
    print(f'{customer_id} may like:')
    
    recommended_restaurants = []
    for name in recommendations:
        recommended_name = rest_df.index[name[0]]
        recommended_restaurants.append(recommended_name)
        print(f"{recommended_name}, "
              f"cuisine type {restaurant_data['cuisine_type'][recommended_name]}, "
              f"average rating {restaurant_data['average_rating'][recommended_name]}, "
              f"average cost {restaurant_data['average_order_cost'][recommended_name]}")
    
    return recommended_restaurants

def split_data_for_evaluation(filtered_df, test_size=0.2, random_state=42):
    """
    Split data into training and test sets for evaluation.
    
    Args:
        filtered_df (pd.DataFrame): DataFrame with filtered restaurant data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (df_train, df_test) DataFrames for training and testing
    """
    # Group by customer to ensure we keep some restaurants for each customer
    customers = filtered_df['customer_id'].unique()
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    for customer in customers:
        customer_data = filtered_df[filtered_df['customer_id'] == customer]
        
        # Only split if the customer has rated more than one restaurant
        if len(customer_data) > 1:
            customer_train, customer_test = train_test_split(
                customer_data, test_size=test_size, random_state=random_state
            )
            df_train = pd.concat([df_train, customer_train])
            df_test = pd.concat([df_test, customer_test])
        else:
            # If customer only rated one restaurant, keep it in training
            df_train = pd.concat([df_train, customer_data])
    
    return df_train, df_test

def calculate_hit_ratio(df_train, df_test, k=10):
    """
    Calculate Hit Ratio@k for both restaurant-based and customer-based recommendations.
    
    Args:
        df_train (pd.DataFrame): Training DataFrame
        df_test (pd.DataFrame): Test DataFrame
        k (int): Number of recommendations to generate for evaluation
        
    Returns:
        tuple: (restaurant_hit_ratio, customer_hit_ratio) Hit ratio scores
    """
    # Create metadata and matrix for training data
    restaurant_data = create_restaurant_metadata(df_train)
    rest_df = create_restaurant_customer_matrix(df_train)
    
    # For restaurant-based recommendation evaluation
    restaurant_hits = 0
    restaurant_total = 0
    
    # For customer-based recommendation evaluation
    customer_hits = 0
    customer_total = 0
    
    # Get unique customers and restaurants in test set
    test_customers = df_test['customer_id'].unique()
    
    # Evaluate restaurant-based recommendations
    for customer in test_customers:
        customer_test_restaurants = set(df_test[df_test['customer_id'] == customer]['restaurant_name'])
        customer_train_restaurants = set(df_train[df_train['customer_id'] == customer]['restaurant_name'])
        
        # Skip if customer has no restaurants in train or test
        if not customer_train_restaurants or not customer_test_restaurants:
            continue
        
        # For each restaurant the customer liked in training, recommend similar restaurants
        for restaurant in customer_train_restaurants:
            if restaurant in rest_df.index:
                recommendations = recommend_restaurant(rest_df, restaurant, k)
                rec_restaurant_names = [rest_df.index[rec[0]] for rec in recommendations]
                
                # Check if any recommended restaurant is in the test set for this customer
                if any(rec in customer_test_restaurants for rec in rec_restaurant_names):
                    restaurant_hits += 1
                restaurant_total += 1
        
        # Customer-based recommendation (using their top restaurant)
        if len(customer_train_restaurants) > 0:
            favorite_restaurant = df_train[
                (df_train['customer_id'] == customer)
            ].sort_values(by='rating', ascending=False)['restaurant_name'].values
            
            if len(favorite_restaurant) > 0:
                favorite_restaurant = favorite_restaurant[0]
                if favorite_restaurant in rest_df.index:
                    recommendations = recommend_restaurant(rest_df, favorite_restaurant, k)
                    rec_restaurant_names = [rest_df.index[rec[0]] for rec in recommendations]
                    
                    # Check if any recommended restaurant is in the test set for this customer
                    if any(rec in customer_test_restaurants for rec in rec_restaurant_names):
                        customer_hits += 1
                    customer_total += 1
    
    # Calculate hit ratios
    restaurant_hit_ratio = restaurant_hits / restaurant_total if restaurant_total > 0 else 0
    customer_hit_ratio = customer_hits / customer_total if customer_total > 0 else 0
    
    return restaurant_hit_ratio, customer_hit_ratio

def main():
    """
    Main function to run the restaurant recommendation system.
    """
    # Path to the dataset
    path = "/Users/vihaannama/Documents/Personal-Projects/nyc-food-recommender/data/food_order.csv"
    
    # Load and preprocess data
    df = load_and_preprocess_data(path)
    
    # Filter popular restaurants
    filtered_df, restaurant_list = filter_popular_restaurants(df, min_ratings=40)
    
    # Create restaurant metadata
    restaurant_data = create_restaurant_metadata(filtered_df)
    
    # Create restaurant-customer matrix
    rest_df = create_restaurant_customer_matrix(filtered_df)
    
    # Split data for evaluation
    df_train, df_test = split_data_for_evaluation(filtered_df)
    
    print("Data split for evaluation:")
    print(f"Training set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    
    # Calculate hit ratios
    restaurant_hit_ratio, customer_hit_ratio = calculate_hit_ratio(df_train, df_test, k=10)
    
    print("\nEvaluation Results:")
    print(f"Restaurant-based Recommendation Hit Ratio@10: {restaurant_hit_ratio:.4f}")
    print(f"Customer-based Recommendation Hit Ratio@10: {customer_hit_ratio:.4f}")
    
    # Example of restaurant-based recommendation
    my_fav_restaurant = np.random.choice(restaurant_list)
    print("\nExample of Restaurant-based Recommendation:")
    recomended_restaurants = recommend_restaurant(rest_df, my_fav_restaurant, 3)
    display_restaurant_recommendations(rest_df, restaurant_data, my_fav_restaurant, recomended_restaurants)
    
    # Example of customer-based recommendation
    print("\nExample of Customer-based Recommendation:")
    customer_list = filtered_df['customer_id'].unique()
    John_Doe = np.random.choice(customer_list)
    get_customer_recommendations(filtered_df, rest_df, restaurant_data, John_Doe, 3)

if __name__ == "__main__":
    main()