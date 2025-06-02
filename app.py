import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('gbmodel.pkl')


# Define the input options
towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
         'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
         'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
         'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
         'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
         'TOA PAYOH', 'WOODLANDS', 'YISHUN']

flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM',
              'MULTI-GENERATION']

storey_ranges = ['10 TO 12', '01 TO 03', '04 TO 06', '07 TO 09', '13 TO 15',
                 '19 TO 21', '22 TO 24', '16 TO 18', '34 TO 36', '28 TO 30',
                 '37 TO 39', '49 TO 51', '25 TO 27', '40 TO 42', '31 TO 33',
                 '46 TO 48', '43 TO 45']



# Streamlit app
st.title("HDB Resale Price Prediction")

# User inputs
selected_town = st.selectbox("Select Town", towns)
selected_flat_type = st.selectbox("Select Flat Type", flat_types)
selected_storey_range = st.selectbox("Select Storey Range", storey_ranges)
selected_floor_area_sqm = st.slider("Floor Area (sqm)", min_value=30, max_value=200, value=70)

# Predict button
if st.button("Predict Price"):
    # Create a dictionary for the input features
    input_data = {
        'town': selected_town,
        'flat_type': selected_flat_type,
        'storey_range': selected_storey_range,
        'floor_area_sqm': selected_floor_area_sqm
    }
    
    # Convert input data to a DataFrame and one-hot encode
    input_df= pd.DataFrame({'town': [selected_town],
                            'flat_type': [selected_flat_type],
                            'storey_range': [selected_storey_range],
                            'floor_area_sqm': [selected_floor_area_sqm]})
    input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)


    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Resale Price: ${prediction:,.2f}")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)