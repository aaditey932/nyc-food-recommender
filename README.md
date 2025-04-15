# 🍽️ NYC Food Recommendation System

## A Personalized Recommendation System for Food Delivery in New York City

This project builds a recommendation system that suggests restaurants based on customer behavior and restaurant attributes. It implements and compares three different approaches: a simple rule-based model, classical machine learning models, and a deep learning neural network, in order to evaluate effectiveness and performance.

---

## 📊 Dataset

The dataset used in this project is sourced from food ordering history in NYC and can be found [here](https://www.kaggle.com/datasets/ahsan81/food-ordering-and-delivery-app-dataset). It contains:

- Customer IDs  
- Restaurant names  
- Cuisine types  
- Order preparation times  
- Delivery times  
- Ratings  

---

## 🧪 Approaches Implemented 

### [`models/`](models)
Contains model checkpoint `.pkl` files along with training and inference scripts.

---

### ✅ 1. **Naïve Approach**

A basic rule-based model that filters restaurants by maximum wait time and recommends the one with the highest average rating.

📄 Implemented in: `models/naive/naïve.py`

---

### ✅ 2. **Classical ML Approach**

This method explores **Collaborative Filtering**, implementing both:

- **Restaurant-Based Filtering**: Finds restaurants similar to one you already like by identifying rating similarities across users.  
- **Customer-Based Filtering**: Learns your preferences and recommends restaurants favored by similar users.

Also includes optional filtering by prep time and delivery constraints.

📄 Implemented in: `models/traditional/ml_eval.py`

#### Restaurant-Based vs. Customer-Based Recommendation Approaches

**Restaurant-Based Approach**  
Finds similar restaurants based on one you already like. The system uses collaborative filtering to identify rating patterns across customers.  
🧠 Think: *“People who like X also like Y.”*

**Customer-Based Approach**  
Personalized to your profile. It identifies your favorite restaurant and recommends similar ones.  
🧠 Think: *“Based on your taste, you’ll love these.”*

Both approaches filter results using practical constraints like delivery or prep time.

---

### ✅ 3. **Deep Learning Approach**

A neural network model implemented in **PyTorch** using the following structure:

- **User input**: History of restaurant visits, encoded into embeddings  
- **Restaurant input**: Fused representation of attributes (cuisine, prep time, delivery time)  
- **Model**: Fully connected layers to learn interactions and predict ratings  
- **Output**: A personalized score predicting the user's rating for each restaurant  

📄 Notebook: `models/deep/ncf_from_scratch.py`

---

## 🤖 Deep Learning Approach: Neural Collaborative Filtering (NCF)

This module implements a *Neural Collaborative Filtering (NCF)* model to generate personalized restaurant recommendations.

### 🔍 Objective

Predict which restaurants a user is most likely to order from using deep learning that combines embeddings, MLPs, and contextual features.

### 📥 Inputs Used

| Column               | Description                                    |
|----------------------|------------------------------------------------|
| customer_id          | Unique user ID, encoded as `user` index       |
| restaurant_name      | Encoded as `item` index                        |
| rating               | Filtered for valid ratings                     |
| delivery_time        | Normalized                                     |
| food_preparation_time | Normalized                                    |
| order_id             | Used to sort historical visits                 |
| past_restaurants     | Encoded and padded sequence of past visits     |

### 🏗️ Model Architecture

- **GMF (Generalized Matrix Factorization)**  
- **MLP (Multi-Layer Perceptron)**  
- **Contextual features** (delivery/prep time)  
- **NeuMF fusion layer**

### 🏋️‍♂️ Training Details

- Loss: Binary crossentropy  
- Optimizer: Adam (`lr=0.001`)  
- Batch size: 128  
- Epochs: 5  
- Negative sampling: 4 negatives per positive

---

## 📈 Evaluation Metrics

### 🎯 Hit Ratio@10

> Measures how often the **true restaurant** appears in the **top 10** predictions.

**Process:**
1. For each user, sample **99 negative restaurants + 1 true one**  
2. Model predicts scores for all 100  
3. If the true item is ranked in top 10 → it's a hit

This reflects the **ranking quality** of the system.

---

### 🔢 Evaluation Results

| Method                | Hit Ratio@10 |
|-----------------------|--------------|
| Naïve                 | 0.016        |
| ML - Restaurant Based | 0.2279       |
| ML - Customer Based   | 0.1804       |
| Deep Learning         | **0.52**     |

---

## 📦 Comprehensive Analysis of All Recommendation Approaches

### 1. Restaurant-Based Approach

**Model:**
- Cosine similarity between restaurants
- Restaurant-by-customer pivot table
- Missing ratings filled with 0s

**Pipeline:**
1. Preprocess dataset  
2. Filter popular restaurants (ratings > 40)  
3. Create pivot table  
4. Compute cosine similarities  
5. Sort & return top-N  
6. Filter by max order time

---

### 2. Customer-Based Approach

**Model:**
- Hybrid approach based on user history
- Finds favorite restaurant first  
- Reuses restaurant similarity model

**Pipeline:**
1. Preprocess dataset  
2. Identify user's top-rated restaurant  
3. Use it to retrieve similar restaurants  
4. Apply filters (e.g., order time)  
5. Return personalized results

---

### 3. Best Rated Restaurant (Simple Recommendation)

**Model:**
- Sort restaurants by average rating
- Filters applied: order time, prep time

**Pipeline:**
1. Preprocess dataset  
2. Compute average ratings  
3. Filter by time constraints  
4. Return highest-rated restaurant

---

## 🧠 Industry Context

### Previous Recommendation Systems in NYC Food Service

#### Traditional:

**Yelp**
- Collaborative filtering based on reviews  
- Geo-based clustering  
- Filters: cuisine, price, location

**Seamless/Grubhub**
- Started with item-based CF  
- Later integrated delivery constraints  
- Built NYC neighborhood-specific models

**Local NYC Apps**
- Content-based filtering (cuisine + distance)  
- Few personalization features

#### Advanced:

**UberEats**
- Contextual multi-armed bandits  
- Time-of-day awareness  
- Deep learning for prep time prediction

**DoorDash**
- Neural networks + time components  
- Traffic/weather-based ETA adjustments  
- Personalized offers

**NYC Startups**
- Hybrid CF + content-based systems  
- Some use social network graphs  
- RL-based reward models for local optimization

---

## 🖥️ Web App

A working **Streamlit app** is included to demonstrate the full recommendation pipeline interactively.

📍 Hosted version: [Click to open](https://impatient-taster.streamlit.app/)

---

## 🧪 How to Run Locally

### 1. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run app.py
