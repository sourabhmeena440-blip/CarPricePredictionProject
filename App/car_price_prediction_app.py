import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

def formatted_values(value): 
    if value >= 10000000:
        return f"{value/10000000:.2f} Crores"
    
    elif value >= 100000:
        return f"{value/100000:.2f} Lakhs"
    
    elif value >=1000:
        return f"{value/1000:.2f} K"
    
    else:
        return str(int(value))

@st.cache_resource
def load_model():
    with open("saved_models/RandomForestRegressor.pkl", "rb") as f:
        model = pickle.load(f)

    with open("saved_scaling/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()


st.set_page_config(page_title="Car Price Prediction",
 page_icon="🚗",
 layout="wide"

)
st.title("Car Price Prediction 🚗")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    car_name = st.text_input("Car Name")
    vehicle_age = st.number_input("Vehicle Age", min_value=0,step=1)
    km_driven = st.number_input("Km driven", min_value=0,step=100)
    mileage = st.number_input("Mileage", min_value=0.0,step=0.1)



with col2:
    engine = st.number_input("Engine (cc)", min_value=0,step=100)
    max_power = st.number_input("Max Power", min_value=0.0,step=1.0)
    seats = st.number_input("Seats", min_value=0,step=1, max_value=7)

st.markdown("### Vehicle Details")

seller_type= st.radio(
    "Seller Type",
    ["Dealer", "Individual", "Trustmark Dealer"],
    horizontal=True
)

fuel_type= st.radio(
    "Fuel Type",
    ["CNG", "Diesel", "Electric", "LPG", "Petrol"],
    horizontal=True
)

transmission_type= st.radio(
    "Transmission Type",
    ["Automatic", "Manual"],
    horizontal=True
)

if st.button("Predict Price", use_container_width=True):
    input_values =[]

    # numeric Features
    input_values.append(int(vehicle_age))
    input_values.append(int(km_driven))
    input_values.append(float(mileage))
    input_values.append(int(engine))
    input_values.append(float(max_power))
    input_values.append(int(seats))

    # seller type encodig

    seller_dict={
        "Dealer":[1,0,0],
        "Individual":[0,1,0],
        "Trustmark Dealer":[0,0,1]
    }

    input_values.extend(seller_dict[seller_type])

    # fuel type encodig

    fuel_dict={
        "CNG":[1,0,0,0,0],
        "Diesel":[0,1,0,0,0],
        "Electric":[0,0,1,0,0],
        "LPG":[0,0,0,1,0],
        "Petrol":[0,0,0,0,1]
    }

    input_values.extend(fuel_dict[fuel_type])
    
    # transmission type encodig

    transmission_dict={
        "Automatic":[1,0],
        "Manual":[0,1]
    }

    input_values.extend(transmission_dict[transmission_type])
   # print(input_values)

try:
    import numpy as np

    # Convert list to 2D array
    input_array = np.array([input_values])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(
        f"💰 Predicted Price: Rs. {formatted_values(prediction)}"
    )

    #with st.expander("Model Input Features"):
    #   st.write(input_values)

except Exception as e:
    st.error(f"Error: {e}")


st.markdown("---")
st.caption("🚗 Car Price Prediction App | Developed by Sourabh")



