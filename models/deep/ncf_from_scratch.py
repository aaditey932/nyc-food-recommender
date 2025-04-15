import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import joblib

# Create models/deep directory if it doesn't exist
os.makedirs("models/deep", exist_ok=True)

# Load dataset
df = pd.read_csv("data/food_order.csv")
print("Dataset loaded successfully with", len(df), "records")

# Filter and preprocess ratings
df = df[df["rating"] != "Not given"]
df["rating"] = df["rating"].astype(float)  # Changed to float to handle missing values better

# Encode categorical features
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df["user"] = user_encoder.fit_transform(df["customer_id"])
df["item"] = item_encoder.fit_transform(df["restaurant_name"])

# Save encoders for later use in the app
joblib.dump(user_encoder, "models/deep/user_encoder.pkl")
joblib.dump(item_encoder, "models/deep/item_encoder.pkl")
print("Encoders saved successfully to models/deep")

# Normalize numeric features
delivery_scaler = MinMaxScaler()
prep_scaler = MinMaxScaler()
df["delivery_norm"] = delivery_scaler.fit_transform(df[["delivery_time"]])
df["prep_norm"] = prep_scaler.fit_transform(df[["food_preparation_time"]])

# Save scalers for use in the app
joblib.dump(delivery_scaler, "models/deep/delivery_scaler.pkl")
joblib.dump(prep_scaler, "models/deep/prep_scaler.pkl")
print("Feature scalers saved successfully to models/deep")

# Build user history column: past visited restaurants
user_history = {}
past_restaurants_col = []

print("Building user history...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    user = row["customer_id"]
    if user not in user_history:
        user_history[user] = []
    past_restaurants_col.append(user_history[user].copy())  # append current history
    user_history[user].append(row["item"])  # update history

df["past_restaurants"] = past_restaurants_col

num_users = df["user"].nunique()
num_items = df["item"].nunique()
print(f"Dataset contains {num_users} unique users and {num_items} unique restaurants")

# Sort and split
df = df.sort_values(["user", "order_id"])
test = df.groupby("user").tail(1)
train = df.drop(test.index)
print(f"Training set: {len(train)} samples, Test set: {len(test)} samples")

# Build negative samples for training
def generate_training_data(train_df, num_items, num_neg=4):
    user_input, item_input, delivery_input, prep_input, history_input, labels = [], [], [], [], [], []
    user_item_set = set(zip(train_df.user, train_df.item))

    print("Generating training data with negative samples...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        u, i = row["user"], row["item"]
        user_input.append(u)
        item_input.append(i)
        delivery_input.append(row["delivery_norm"])
        prep_input.append(row["prep_norm"])
        history_input.append(row["past_restaurants"])
        labels.append(1)

        for _ in range(num_neg):
            j = random.randint(0, num_items - 1)
            while (u, j) in user_item_set:
                j = random.randint(0, num_items - 1)
            user_input.append(u)
            item_input.append(j)
            delivery_input.append(train_df["delivery_norm"].mean())
            prep_input.append(train_df["prep_norm"].mean())
            history_input.append(row["past_restaurants"])
            labels.append(0)

    return (
        np.array(user_input),
        np.array(item_input),
        np.array(delivery_input),
        np.array(prep_input),
        history_input,
        np.array(labels)
    )

train_users, train_items, train_delivery_times, train_prep_times, train_histories, train_labels = generate_training_data(train, num_items)

# Define Keras model
def build_ncf_model(num_users, num_items, mf_dim=16, mlp_dim=16, layers=[64, 32, 16]):
    """
    Build a Neural Collaborative Filtering model based on the provided architecture diagram.
    
    Args:
        num_users: Number of unique users in the dataset
        num_items: Number of unique items in the dataset
        mf_dim: Dimension of the Matrix Factorization embedding
        mlp_dim: Dimension of the MLP embedding
        layers: List of layer dimensions for the MLP component
    """
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    delivery_input = Input(shape=(1,), name='delivery_input')
    prep_input = Input(shape=(1,), name='prep_input')
    history_input = Input(shape=(None,), name='history_input', dtype='int32')
    
    # Embedding layers for context features
    history_embed = Embedding(num_items, mlp_dim, name='history_embedding')(history_input)
    history_vec = GlobalAveragePooling1D()(history_embed)
    
    # Process delivery and prep time features
    delivery_dense = Dense(mlp_dim, activation='relu')(delivery_input)
    prep_dense = Dense(mlp_dim, activation='relu')(prep_input)
    
    delivery_vec = Flatten()(delivery_dense)
    prep_vec = Flatten()(prep_dense)
    
    # Matrix Factorization Part (GMF)
    mf_user_embedding = Embedding(num_users, mf_dim, name='mf_user_embedding')(user_input)
    mf_item_embedding = Embedding(num_items, mf_dim, name='mf_item_embedding')(item_input)
    
    mf_user_latent = Flatten()(mf_user_embedding)
    mf_item_latent = Flatten()(mf_item_embedding)
    
    # Element-wise product for GMF
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])
    
    # Non-linear transformation of dot product (GMF Layer)
    gmf_output = Dense(mf_dim, activation='relu', name='gmf_layer')(mf_vector)
    
    # MLP Part
    mlp_user_embedding = Embedding(num_users, mlp_dim, name='mlp_user_embedding')(user_input)
    mlp_item_embedding = Embedding(num_items, mlp_dim, name='mlp_item_embedding')(item_input)
    
    mlp_user_latent = Flatten()(mlp_user_embedding)
    mlp_item_latent = Flatten()(mlp_item_embedding)
    
    # Concatenate user and item vectors for MLP path
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    
    # MLP Layers with ReLU activation
    for i, layer_size in enumerate(layers):
        layer_name = f'mlp_layer_{i+1}'
        mlp_vector = Dense(layer_size, activation='relu', name=layer_name)(mlp_vector)
    
    # Combining context features with MLP path
    context_vector = Concatenate()([history_vec, delivery_vec, prep_vec])
    enhanced_mlp = Concatenate()([mlp_vector, context_vector])
    
    # Concatenate GMF and MLP parts (NeuMF Layer)
    neufm_vector = Concatenate()([gmf_output, enhanced_mlp])
    
    # Final prediction layer
    output = Dense(1, activation='sigmoid', name='prediction')(neufm_vector)
    
    model = Model(
        inputs=[user_input, item_input, delivery_input, prep_input, history_input],
        outputs=output
    )
    model.compile(
        optimizer=Adam(0.001),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

print("Building NCF model...")
model = build_ncf_model(num_users, num_items)
model.summary()

# Pad histories for training
print("Padding sequences...")
padded_histories = pad_sequences(train_histories, maxlen=10, padding='post', truncating='post')

# Train the model
print("Training model...")
model.fit(
    [train_users, train_items, train_delivery_times, train_prep_times, padded_histories],
    train_labels,
    batch_size=128,
    epochs=5,
    verbose=1
)

# Save the model for use in the app
model.save("models/deep/ncf_model.h5")
print("Model saved successfully to models/deep/ncf_model.h5")

# Evaluation using Hit Ratio@10
def hit_ratio_at_k(model, test_df, num_items, k=10):
    hits = 0
    print("Evaluating model with Hit Ratio@10...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        u = row["user"]
        true_item = row["item"]
        delivery = row["delivery_norm"]
        prep = row["prep_norm"]
        history = row["past_restaurants"]

        # Generate 99 negative samples
        negative_items = set()
        while len(negative_items) < 99:
            j = random.randint(0, num_items - 1)
            if j != true_item:
                negative_items.add(j)
        test_items = list(negative_items) + [true_item]

        users = np.full(len(test_items), u)
        deliveries = np.full(len(test_items), delivery)
        preps = np.full(len(test_items), prep)
        histories = pad_sequences([history], maxlen=10, padding='post', truncating='post')
        histories = np.repeat(histories, len(test_items), axis=0)

        predictions = model.predict([users, np.array(test_items), deliveries, preps, histories], verbose=0).flatten()
        top_k_items = np.argsort(predictions)[-k:]
        recommended_items = np.array(test_items)[top_k_items]

        if true_item in recommended_items:
            hits += 1

    return hits / len(test_df)

# Evaluate
hr10 = hit_ratio_at_k(model, test, num_items, k=10)
print(f"\n🎯 Hit Ratio@10: {hr10:.4f}")
print("\nModel training complete! You can now run app.py to use the recommender system.")