import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


from sklearn.metrics.pairwise import cosine_similarity


path = "/Users/vihaannama/Documents/Personal-Projects/nyc-food-recommender/data/food_order.csv"
df = pd.read_csv(path)
df_safe = df.copy()
df.head()

df['total_order_time'] = df['food_preparation_time'] + df['delivery_time']
df.head()

(df['rating'] == 'Not given').sum()

df['rating'] = df['rating'].replace('Not given', None)
df['rating'] = df['rating'].astype('float64')
df.info()

df['rating'] = df.groupby(['restaurant_name'], sort=False)['rating'].apply(lambda x: x.fillna(x.mean())).values
df.info()

df.dropna(inplace=True)

df_ = df.groupby('restaurant_name')['rating'].size().sort_values(ascending=False)
df_ = df_.reset_index()
df_


restaurant_list = df_[df_['rating'] > 40]['restaurant_name'].values
print(len(restaurant_list))


filtered_df = df[df['restaurant_name'].isin(restaurant_list)]
filtered_df



restaurant_data = filtered_df.groupby(['restaurant_name', 'cuisine_type']).aggregate({'rating': 'mean', 'cost_of_the_order': 'mean', 'order_id': 'size'}).reset_index(level='cuisine_type')
restaurant_data = restaurant_data.rename(columns={'rating': 'average_rating', 'cost_of_the_order': 'average_order_cost', 'order_id': 'rating_count'})
restaurant_data[['average_rating', 'average_order_cost']] = restaurant_data[['average_rating', 'average_order_cost']].round(2)
restaurant_data = restaurant_data.to_dict()


rest_df = filtered_df.pivot_table(index='restaurant_name', columns='customer_id', values='rating')
rest_df.fillna(0, inplace=True)
rest_df


def recomend_restaurant(restaurant_df, restaurant_name, restaurant_limit = 1):
    restaurant_index = np.where(restaurant_df.index == my_fav_restaurant)[0][0]
    similarities = cosine_similarity(restaurant_df)
    recomended_restaurants = sorted(list(enumerate(similarities[restaurant_index])), key=lambda x: x[1], reverse=True)[1: restaurant_limit + 1]
    return recomended_restaurants


my_fav_restaurant = np.random.choice(restaurant_list)
filtered_df[filtered_df['restaurant_name'] == my_fav_restaurant].head()



restaurant_limit = 3
recomended_restaurants = recomend_restaurant(rest_df, my_fav_restaurant, restaurant_limit)
print(f"Your favorite restaurant: {my_fav_restaurant}, cuisine type: {restaurant_data['cuisine_type'][my_fav_restaurant]}")
print(f'Customers like {my_fav_restaurant}, also like:')
for name in recomended_restaurants:
    print(rest_df.index[name[0]], f",\
    cuisine type {restaurant_data['cuisine_type'][rest_df.index[name[0]]]},\
    average rating {restaurant_data['average_rating'][rest_df.index[name[0]]]},\
    average cost {restaurant_data['average_order_cost'][rest_df.index[name[0]]]}")



customer_list = filtered_df['customer_id'].values
John_Doe = np.random.choice(customer_list)
John_Doe


favorite_restaurant = filtered_df[filtered_df['customer_id'] == John_Doe].sort_values(by = 'rating', ascending=False)['restaurant_name'].values[0]


restaurant_limit = 3
recomended_restaurants = recomend_restaurant(rest_df, favorite_restaurant, restaurant_limit)
print(f"John_Doe's restaurant, {favorite_restaurant}, they make {restaurant_data['cuisine_type'][favorite_restaurant]} food")
print('John_Doe may like:')
for name in recomended_restaurants:
    print(rest_df.index[name[0]], f",\
    cuisine type {restaurant_data['cuisine_type'][rest_df.index[name[0]]]},\
    average rating {restaurant_data['average_rating'][rest_df.index[name[0]]]},\
    average cost {restaurant_data['average_order_cost'][rest_df.index[name[0]]]}")