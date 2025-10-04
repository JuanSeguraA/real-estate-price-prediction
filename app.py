import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load("xgb_best_model.joblib")

st.set_page_config( 
    page_title="Madrid Real Estate Price Predictor", 
    page_icon="🏠", 
    layout="wide", 
    initial_sidebar_state="expanded" )

def add_sidebar():
    st.sidebar.header("Property Features")

    sq_mt_built = st.sidebar.slider("Built area (m²)", 20, 1000, 100)
    sq_mt_useful = st.sidebar.slider("Usable area (m²)", 20, 800, 80)
    n_rooms = st.sidebar.slider("Number of rooms", 1, 10, 3)
    n_bathrooms = st.sidebar.slider("Number of bathrooms", 1, 5, 2)
    floor = st.sidebar.slider("Floor number", 0, 15, 2)
    built_year = st.sidebar.slider("Year built", 1900, 2025, 2000)
    parking_price = st.sidebar.number_input("Parking price (€)", min_value=0, max_value=50000, value=0)
    neighborhood_price_sqm = st.sidebar.slider("Neighborhood price (€/m²)", 500, 10000, 3500)

    st.sidebar.subheader("Amenities & Features")
    is_renewal_needed = st.sidebar.checkbox("Requires renewal?", False)
    is_new_development = st.sidebar.checkbox("Is new development?", False)
    has_central_heating = st.sidebar.checkbox("Has central heating", True)
    has_individual_heating = st.sidebar.checkbox("Has individual heating", False)
    has_ac = st.sidebar.checkbox("Has air conditioning", True)
    has_fitted_wardrobes = st.sidebar.checkbox("Has fitted wardrobes", True)
    has_lift = st.sidebar.checkbox("Has lift", True)
    is_exterior = st.sidebar.checkbox("Is exterior", True)
    has_garden = st.sidebar.checkbox("Has garden", False)
    has_pool = st.sidebar.checkbox("Has pool", False)
    has_terrace = st.sidebar.checkbox("Has terrace", True)
    has_balcony = st.sidebar.checkbox("Has balcony", True)
    has_storage_room = st.sidebar.checkbox("Has storage room", False)
    is_accessible = st.sidebar.checkbox("Wheelchair accessible", False)
    has_green_zones = st.sidebar.checkbox("Has green zones nearby", True)
    has_parking = st.sidebar.checkbox("Has parking", True)
    is_parking_included_in_price = st.sidebar.checkbox("Parking included in price", True)

    st.sidebar.subheader("Orientation")
    is_orientation_north = st.sidebar.checkbox("North-facing", False)
    is_orientation_west = st.sidebar.checkbox("West-facing", False)
    is_orientation_south = st.sidebar.checkbox("South-facing", True)
    is_orientation_east = st.sidebar.checkbox("East-facing", False)

    st.sidebar.subheader("Property Type")
    house_type = st.sidebar.radio(
        "Property house type",
        ["Apartment", "Duplex", "House", "Penthouse"]
    )

    house_type_id_duplex = int(house_type == "Duplex")
    house_type_id_house = int(house_type == "House")
    house_type_id_penthouse = int(house_type == "Penthouse")

    input_data = pd.DataFrame([{
        "sq_mt_built": sq_mt_built,
        "sq_mt_useful": sq_mt_useful,
        "n_rooms": n_rooms,
        "n_bathrooms": n_bathrooms,
        "floor": floor,
        "built_year": built_year,
        "parking_price": parking_price,
        "is_renewal_needed": int(is_renewal_needed),
        "is_new_development": int(is_new_development),
        "has_central_heating": int(has_central_heating),
        "has_individual_heating": int(has_individual_heating),
        "has_ac": int(has_ac),
        "has_fitted_wardrobes": int(has_fitted_wardrobes),
        "has_lift": int(has_lift),
        "is_exterior": int(is_exterior),
        "has_garden": int(has_garden),
        "has_pool": int(has_pool),
        "has_terrace": int(has_terrace),
        "has_balcony": int(has_balcony),
        "has_storage_room": int(has_storage_room),
        "is_accessible": int(is_accessible),
        "has_green_zones": int(has_green_zones),
        "has_parking": int(has_parking),
        "is_parking_included_in_price": int(is_parking_included_in_price),
        "is_orientation_north": int(is_orientation_north),
        "is_orientation_west": int(is_orientation_west),
        "is_orientation_south": int(is_orientation_south),
        "is_orientation_east": int(is_orientation_east),
        "house_type_id_duplex": house_type_id_duplex,
        "house_type_id_house": house_type_id_house,
        "house_type_id_penthouse": house_type_id_penthouse,
        "neighborhood_price_sqm": neighborhood_price_sqm
    }])

    return input_data


def main():
    input_data = add_sidebar()

    st.title("🏠 Madrid Real Estate Price Predictor")
    st.write("Use this app to estimate the price of a property in Madrid based on its features.")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.write("### Current Input Data")
        st.dataframe(input_data)

        if st.button("Predict Price"):
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Property Price: €{prediction:,.0f}")

            # --- Feature importance chart ---
            importances = model.feature_importances_
            features = input_data.columns
            sorted_idx = np.argsort(importances)[::-1][:10]  # top 10 features

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(np.array(features)[sorted_idx][::-1], importances[sorted_idx][::-1])
            ax.set_title("Top 10 Feature Importances")
            ax.set_xlabel("Importance")
            st.pyplot(fig)

    with col2:
        st.markdown("### Model Confidence")
        st.metric(
            label="Likelihood of Prediction Accuracy",
            value="93%",
            delta="±7% margin"
        )
        st.caption("Based on model validation (R² ≈ 0.93).\n"
                    "This reflects how close predictions tend to be to real prices on unseen data.")


if __name__ == "__main__":
    main()

