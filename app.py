import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# from streamlit_plotly_events import plotly_events # Ensure this is removed or commented out
from utils import (
    generate_users, generate_products, update_user_vector,
    find_similar_products, find_similar_users, record_user_like,
    get_user_likes, get_liked_products_vectors
)

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Vector Search Visualizer")

st.title("🛒 Vector Search Visualizer")
st.write("사용자 선호도에 따라 변화하는 상품 추천과 벡터 공간 시각화")

# --- Constants & Parameters ---
N_FEATURES = 3 # Using 3 features for 3D visualization
LEARNING_RATE = 1.0 # Reduced learning rate to prevent vector going too far off-screen
N_RECOMMENDATIONS = 10
N_SIMILAR_USERS = 5

# --- Session State Initialization ---
def init_session_state(force_reset=False):
    defaults = {
        'users_df': lambda: generate_users(n_users=10, n_features=N_FEATURES),
        'products_df': lambda: generate_products(n_products=300, n_features=N_FEATURES),
        'selected_user_id': lambda: st.session_state.users_df['user_id'].iloc[0] if 'users_df' in st.session_state and not st.session_state.users_df.empty else None,
        'show_similar_users': False,
        'previous_user_vector': None, # Track previous vector for visualization
        'last_liked_product_id': None, # Track last liked product for highlight
        'highlight_recommendations': False, # Track whether to highlight recommendations
        'recommendation_page': 1 # Current page for recommendations
    }
    for key, default_value in defaults.items():
        if key not in st.session_state or force_reset:
            st.session_state[key] = default_value() if callable(default_value) else default_value

    # Ensure selected_user_id is valid after potential reset
    if 'users_df' in st.session_state and not st.session_state.users_df.empty:
         if st.session_state.selected_user_id not in st.session_state.users_df['user_id'].tolist():
             st.session_state.selected_user_id = st.session_state.users_df['user_id'].iloc[0]

# Initial call or reset if needed
if 'users_df' not in st.session_state:
    init_session_state(force_reset=True)

# --- Helper Functions ---
def get_user_vector(user_id):
    if user_id is None or user_id not in st.session_state.users_df['user_id'].tolist():
        return None # Handle cases where user_id might be invalid temporarily
    return st.session_state.users_df.loc[st.session_state.users_df['user_id'] == user_id, 'vector'].values[0]

def get_product_vector(product_id):
    if product_id is None or product_id not in st.session_state.products_df['product_id'].tolist():
        return None
    return st.session_state.products_df.loc[st.session_state.products_df['product_id'] == product_id, 'vector'].values[0]

# Function to reset interaction states (prev vector, last like)
def reset_interaction_state():
    st.session_state.previous_user_vector = None
    st.session_state.last_liked_product_id = None

# --- Refactored Like Logic ---
def handle_like_action(user_id, product_id):
    """Handles the logic when a user likes a product."""
    current_user_vector = get_user_vector(user_id)
    product_vector = get_product_vector(product_id)

    if current_user_vector is not None and product_vector is not None:
        # Store current vector as previous
        st.session_state.previous_user_vector = current_user_vector.copy()
        # Store liked product ID
        st.session_state.last_liked_product_id = product_id

        # Update user vector
        new_user_vector = update_user_vector(current_user_vector, product_vector, learning_rate=LEARNING_RATE)
        # Update DataFrame in session state
        try:
            user_idx = st.session_state.users_df.index[st.session_state.users_df['user_id'] == user_id].tolist()[0]
            # Ensure the column exists and update (using .loc is generally safe)
            st.session_state.users_df.loc[user_idx, 'vector'] = new_user_vector
        except IndexError:
            st.error(f"Failed to find user index for {user_id} during like action.")
            return # Stop if user index not found

        # Record the like (using the function from utils)
        record_user_like(user_id, product_id)
        # Rerun to update plot and recommendations
        st.rerun()
    else:
        st.warning(f"Could not process like for User {user_id} and Product {product_id}. Vectors might be missing.")

# --- Sidebar ---
st.sidebar.header("⚙️ Controls")

# User Selection
available_user_ids = st.session_state.users_df['user_id'].tolist() if 'users_df' in st.session_state else []
if available_user_ids:
    try:
        current_index = available_user_ids.index(st.session_state.selected_user_id)
    except ValueError:
        # If selected user somehow doesn't exist (e.g., after regen), default to first
        st.session_state.selected_user_id = available_user_ids[0]
        current_index = 0

    new_selected_user_id = st.sidebar.selectbox(
        "Select User:",
        available_user_ids,
        index=current_index
    )
    # Reset interaction state if user changes
    if new_selected_user_id != st.session_state.selected_user_id:
        st.session_state.selected_user_id = new_selected_user_id
        reset_interaction_state()
        st.rerun() # Rerun to reflect change immediately
else:
    st.sidebar.warning("No users generated yet.")
    st.stop() # Stop execution if no users

selected_user_id = st.session_state.selected_user_id # Use the potentially updated value

st.sidebar.subheader("Actions")
# Button to regenerate data
if st.sidebar.button("🔄 Regenerate Data"):
    init_session_state(force_reset=True)
    # Reset likes when regenerating data
    from utils import USER_PREFERENCES
    USER_PREFERENCES.clear()
    reset_interaction_state() # Also reset interaction states
    st.rerun()

# Toggle similar users view
new_show_similar = st.sidebar.checkbox(
    "Show Similar Users", value=st.session_state.show_similar_users
)
if new_show_similar != st.session_state.show_similar_users:
    st.session_state.show_similar_users = new_show_similar
    st.rerun() # Rerun if the state changes

# Checkbox to highlight recommendations
new_highlight_recs = st.sidebar.checkbox(
    "⭐ 추천 상품 하이라이트", value=st.session_state.highlight_recommendations
)
if new_highlight_recs != st.session_state.highlight_recommendations:
    st.session_state.highlight_recommendations = new_highlight_recs
    st.rerun()

st.sidebar.divider() # Add a divider for visual separation
st.sidebar.subheader("👍 Like by ID")
like_product_id_input = st.sidebar.text_input("Enter Product ID:", key="like_id_input")
if st.sidebar.button("Like ID", key="like_id_button"):
    product_id_to_like = like_product_id_input.strip()
    
    # 숫자만 입력된 경우 'product_' prefix 자동 추가
    if product_id_to_like.isdigit():
        product_id_to_like = f"product_{product_id_to_like}"
        
    if not product_id_to_like:
        st.sidebar.error("Please enter a Product ID.")
    elif product_id_to_like not in st.session_state.products_df['product_id'].tolist():
        st.sidebar.error(f"Product ID '{product_id_to_like}' not found.")
    else:
        st.sidebar.info(f"Attempting to like Product ID: {product_id_to_like}")
        handle_like_action(selected_user_id, product_id_to_like)

# --- Main Layout ---
col1, col2 = st.columns([3, 1]) # Visualization column, Recommendations column

with col1:
    st.header("🌐 Vector Space (3D)")

    # --- Data Preparation for Plot ---
    user_vector = get_user_vector(selected_user_id)
    if user_vector is None:
        st.warning(f"Could not find vector for user {selected_user_id}. Please regenerate data or select another user.")
        st.stop()

    products_df = st.session_state.products_df
    users_df = st.session_state.users_df
    product_vectors = np.stack(products_df['vector'].values)

    # Find liked products for the current user
    liked_product_ids = get_user_likes(selected_user_id)
    liked_products_df = products_df[products_df['product_id'].isin(liked_product_ids)]

    # --- Plotly 3D Scatter Plot (Restored with all features) ---
    fig = go.Figure()

    # 1. Plot all products
    product_rgb_colors = [f'rgb({int(v[0]*255)}, {int(v[1]*255)}, {int(v[2]*255)})' for v in product_vectors]
    product_sizes = [5] * len(products_df)

    # Highlight last liked product if it exists
    if st.session_state.last_liked_product_id:
        try:
            product_id_list = products_df['product_id'].tolist()
            last_liked_index = product_id_list.index(st.session_state.last_liked_product_id)
            product_rgb_colors[last_liked_index] = 'lime'
            product_sizes[last_liked_index] = 10
        except ValueError:
            st.warning("Could not highlight last liked product.")

    # Highlight recommended products if enabled
    if st.session_state.highlight_recommendations:
        recommended_products = find_similar_products(user_vector, products_df, n=N_RECOMMENDATIONS)
        for product_id in recommended_products['product_id']:
            try:
                rec_index = product_id_list.index(product_id)
                product_rgb_colors[rec_index] = 'yellow'
                product_sizes[rec_index] = 8
            except ValueError:
                continue

    # Add all products scatter plot
    fig.add_trace(go.Scatter3d(
        x=product_vectors[:, 0],
        y=product_vectors[:, 1],
        z=product_vectors[:, 2],
        mode='markers',
        marker=dict(
            size=product_sizes,
            color=product_rgb_colors,
            opacity=0.6
        ),
        name='Products',
        text=products_df['product_id'],
        hoverinfo='text'
    ))

    # 2. Plot user vector
    fig.add_trace(go.Scatter3d(
        x=[user_vector[0]],
        y=[user_vector[1]],
        z=[user_vector[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name=f'User {selected_user_id}',
        text=[f'User {selected_user_id}'],
        hoverinfo='text'
    ))

    # 3. Plot similar users if enabled
    if st.session_state.show_similar_users:
        similar_users = find_similar_users(selected_user_id, users_df, n=N_SIMILAR_USERS)
        similar_user_vectors = np.stack(similar_users['vector'].values)
        
        fig.add_trace(go.Scatter3d(
            x=similar_user_vectors[:, 0],
            y=similar_user_vectors[:, 1],
            z=similar_user_vectors[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='purple',
                symbol='diamond',
                opacity=0.7
            ),
            name='Similar Users',
            text=similar_users['user_id'],
            hoverinfo='text'
        ))

    # 4. Plot previous vector if it exists (to show movement)
    if st.session_state.previous_user_vector is not None:
        prev_vector = st.session_state.previous_user_vector
        fig.add_trace(go.Scatter3d(
            x=[prev_vector[0]],
            y=[prev_vector[1]],
            z=[prev_vector[2]],
            mode='markers',
            marker=dict(
                size=15,
                color='rgba(255, 0, 0, 0.3)',  # Transparent red
                symbol='diamond'
            ),
            name='Previous Position',
            text=['Previous Position'],
            hoverinfo='text'
        ))

        # Add arrow to show movement
        fig.add_trace(go.Scatter3d(
            x=[prev_vector[0], user_vector[0]],
            y=[prev_vector[1], user_vector[1]],
            z=[prev_vector[2], user_vector[2]],
            mode='lines',
            line=dict(color='red', width=2),
            name='Movement',
            showlegend=False
        ))

    # --- Layout Settings for Plot ---
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),  # 모든 마진을 0으로 설정
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            zaxis_range=[0, 1],
            aspectmode='cube',  # 정육면체 비율 유지
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)  # 카메라 위치 조정
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"  # 범례 배경 약간 투명하게
        ),
        height=800  # 그래프 높이 증가
    )

    # --- Use st.plotly_chart for rendering ---
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🎨 Recommended Colors")
    
    # Get recommendations
    recommended_products = find_similar_products(user_vector, products_df, n=N_RECOMMENDATIONS)
    
    # Display recommendations
    for _, product in recommended_products.iterrows():
        product_id = product['product_id']
        vector = product['vector']
        
        # Create color box
        color = f'rgb({int(vector[0]*255)}, {int(vector[1]*255)}, {int(vector[2]*255)})'
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                width: 100%;
                height: 50px;
                margin: 5px 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: {'white' if sum(vector) < 1.5 else 'black'};
                border-radius: 5px;
                cursor: pointer;
            ">
                {product_id}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Like button for each recommendation
        if st.button(f"👍 Like", key=f"like_{product_id}"):
            handle_like_action(selected_user_id, product_id)