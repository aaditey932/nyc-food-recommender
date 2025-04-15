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
### [model](models) include pkl files and scripts

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

---

### ✅ 3. **Deep Learning Approach**

A neural network model implemented in **PyTorch** using the following structure:

- **User input**: History of restaurant visits, encoded into embeddings  
- **Restaurant input**: Fused representation of attributes (cuisine, prep time, delivery time)  
- **Model**: Fully connected layers to learn interactions and predict ratings  
- **Output**: A personalized score predicting the user's rating for each restaurant  

🧠 Notebook: [`models/deep/ncf_from_scratch.py`]

---


## 🖥️ Web App

A working **Streamlit app** is included to demonstrate the full recommendation pipeline interactively.

📍 Hosted version: [Click to open](https://impatient-taster.streamlit.app/)

*Note: Replace the link above with your actual deployment URL.*

---

## 🧪 How to Run Locally

### 1. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows


#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

