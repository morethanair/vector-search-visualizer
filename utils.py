import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm # Import norm for Euclidean distance

# --- Data Generation ---

def generate_users(n_users: int, n_features: int = 3) -> pd.DataFrame:
    """Generate random user data and their corresponding vectors."""
    users = []
    for i in range(n_users):
        user_id = f"user_{i}"
        # Example features: age (normalized), purchasing power (normalized), etc.
        features = np.random.rand(n_features)
        users.append({"user_id": user_id, "vector": features})
    return pd.DataFrame(users)

def generate_products(n_products: int, n_features: int = 3) -> pd.DataFrame:
    """Generate random product data and their corresponding vectors. Color (hex) is derived from vector."""
    products = []
    for i in range(n_products):
        product_id = f"product_{i}"
        features = np.random.rand(n_features)

        # Derive hex color directly from the first 3 features (vector components)
        if n_features >= 3:
            r = int(features[0] * 255)
            g = int(features[1] * 255)
            b = int(features[2] * 255)
            # Ensure values are within 0-255 range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            # Convert to hex format #RRGGBB
            derived_hex_color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            derived_hex_color = "#808080" # Default grey in hex

        metadata = {
            "color": derived_hex_color, # Use vector-derived hex color
            "shape": np.random.choice(["Sphere", "Cube", "Pyramid", "Complex"]),
            "size": np.random.choice(["Small", "Medium", "Large"])
        }
        products.append({"product_id": product_id, "vector": features, **metadata})
    return pd.DataFrame(products)

# --- Vector Operations ---

def update_user_vector(user_vector: np.ndarray, liked_product_vector: np.ndarray, learning_rate: float = 0.9) -> np.ndarray:
    """Move the user vector 90% of the way towards the liked product vector."""
    # Calculate the new position as 10% current position + 90% target position
    new_vector = 0.1 * user_vector + 0.9 * liked_product_vector
    # The previous linear interpolation is replaced:
    # new_vector = user_vector + learning_rate * (liked_product_vector - user_vector)
    return new_vector

def find_similar_items(target_vector: np.ndarray, item_vectors: np.ndarray, item_ids: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Find items with vectors closest to the target vector using Euclidean distance."""
    if not item_vectors.size or target_vector.size != item_vectors.shape[1]:
        return pd.DataFrame(columns=['item_id', 'distance'])

    # Calculate Euclidean distances between the target vector and all item vectors
    distances = norm(item_vectors - target_vector, axis=1)

    distance_df = pd.DataFrame({'item_id': item_ids, 'distance': distances})
    # Sort by distance ASCENDING (closer is better) and return top N
    top_similar = distance_df.sort_values(by='distance', ascending=True).head(top_n)
    return top_similar

def find_similar_products(user_vector: np.ndarray, products_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find products closest to the user's vector based on Euclidean distance."""
    product_vectors = np.stack(products_df['vector'].values)
    # Now receives a DataFrame with 'item_id' and 'distance'
    similar_products = find_similar_items(user_vector, product_vectors, products_df['product_id'], top_n)
    # Join with original product data, matching product_id with item_id
    # Sort by distance ASCENDING
    return products_df.merge(similar_products, left_on='product_id', right_on='item_id').sort_values(by='distance', ascending=True)


def find_similar_users(target_user_vector: np.ndarray, users_df: pd.DataFrame, target_user_id: str, top_n: int = 5) -> pd.DataFrame:
    """Find users closest to the target user based on Euclidean distance."""
    # Exclude the target user from the comparison set
    other_users_df = users_df[users_df['user_id'] != target_user_id].copy()
    if other_users_df.empty:
        return pd.DataFrame(columns=['user_id', 'distance'])

    other_user_vectors = np.stack(other_users_df['vector'].values)
    # Use the updated find_similar_items function
    similar_users = find_similar_items(target_user_vector, other_user_vectors, other_users_df['user_id'], top_n)
    # similar_users DataFrame now contains 'item_id' and 'distance'
    # Rename item_id to user_id for clarity
    return similar_users.rename(columns={'item_id': 'user_id'})

# --- Data Persistence for User Preferences (Simple Example) ---
# In a real application, this would involve a database or more robust storage.

USER_PREFERENCES = {} # Dictionary to store liked products for each user

def record_user_like(user_id: str, product_id: str):
    """Record that a user liked a specific product."""
    if user_id not in USER_PREFERENCES:
        USER_PREFERENCES[user_id] = set()
    USER_PREFERENCES[user_id].add(product_id)

def get_user_likes(user_id: str) -> set:
    """Get the set of product IDs liked by a user."""
    return USER_PREFERENCES.get(user_id, set())

def get_liked_products_vectors(user_id: str, products_df: pd.DataFrame) -> np.ndarray:
    """Get the vectors of products liked by a user."""
    liked_product_ids = get_user_likes(user_id)
    if not liked_product_ids:
        return np.array([]) # Return empty array if no likes yet

    liked_products_df = products_df[products_df['product_id'].isin(liked_product_ids)]
    if liked_products_df.empty:
        return np.array([])

    return np.stack(liked_products_df['vector'].values)