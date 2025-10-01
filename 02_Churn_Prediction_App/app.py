import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Define a function for loading files to avoid reloading ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # --- Data Cleaning Step ---
    # This makes our app robust and self-sufficient.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    # Ensure Churn column is numeric for calculations
    # This handles both the original text and if it's already 0/1
    if df['Churn'].dtype == 'object':
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# --- Load Model and Data ---
try:
    df_cleaned = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    model = load_model('churn_model.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure 'churn_model.pkl' and 'WA_Fn-UseC_-Telco-Customer-Churn.csv' are in the same folder.")
    st.stop()

# --- Pre-calculate the columns our model expects ---
df_for_encoding = df_cleaned.drop(['customerID', 'Churn'], axis=1, errors='ignore')
model_columns = pd.get_dummies(df_for_encoding, drop_first=True).columns


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Churn Prediction Tool", "Churn Analysis Dashboard"])
st.sidebar.markdown("---")
st.sidebar.write("Project by an intern.")


# =====================================================================================
# --- Page 1: Prediction Tool ---
# =====================================================================================
if page == "Churn Prediction Tool":
    st.title('Live Customer Churn Prediction')
    st.markdown("Enter a customer's details below to get a real-time churn prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider('Tenure (months)', min_value=0, max_value=72, value=12, help="How many months the customer has been with the company.")
        contract = st.selectbox('Contract', options=df_cleaned['Contract'].unique())
        payment_method = st.selectbox('Payment Method', options=df_cleaned['PaymentMethod'].unique())
        paperless_billing = st.radio('Paperless Billing', options=['Yes', 'No'])

    with col2:
        st.subheader("Charges")
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=float(df_cleaned['TotalCharges'].median()), step=10.0)
        
        st.subheader("Customer Info")
        gender = st.radio('Gender', options=['Male', 'Female'])
        senior_citizen = st.radio('Senior Citizen', options=['No', 'Yes'])
        partner = st.radio('Has a Partner?', options=['Yes', 'No'])
        dependents = st.radio('Has Dependents?', options=['Yes', 'No'])

    with col3:
        st.subheader("Subscribed Services")
        phone_service = st.selectbox('Phone Service', options=['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', options=df_cleaned['MultipleLines'].unique())
        internet_service = st.selectbox('Internet Service', options=df_cleaned['InternetService'].unique())
        online_security = st.selectbox('Online Security', options=df_cleaned['OnlineSecurity'].unique())
        online_backup = st.selectbox('Online Backup', options=df_cleaned['OnlineBackup'].unique())
        device_protection = st.selectbox('Device Protection', options=df_cleaned['DeviceProtection'].unique())
        tech_support = st.selectbox('Tech Support', options=df_cleaned['TechSupport'].unique())
        streaming_tv = st.selectbox('Streaming TV', options=df_cleaned['StreamingTV'].unique())
        streaming_movies = st.selectbox('Streaming Movies', options=df_cleaned['StreamingMovies'].unique())


    if st.button('Predict Churn', type="primary"):
        input_data = {
            'gender': gender, 'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
            'MultipleLines': multiple_lines, 'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 'Contract': contract, 
            'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        input_df = pd.DataFrame([input_data])
        input_df_encoded = pd.get_dummies(input_df, drop_first=True)
        final_input_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        prediction_proba = model.predict_proba(final_input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")
        if prediction_proba > 0.5:
            st.error(f'Prediction: **Churn** (Probability: {prediction_proba:.2%})')
            st.warning("Recommendation: This customer is at high risk of churning. Proactive engagement is recommended.")
        else:
            st.success(f'Prediction: **No Churn** (Probability of Churn: {prediction_proba:.2%})')
            st.info("Recommendation: This customer is likely to stay. Continue standard engagement.")

# =====================================================================================
# --- Page 2: Analysis Dashboard ---
# =====================================================================================
elif page == "Churn Analysis Dashboard":
    st.title('Churn Analysis Dashboard')
    st.markdown("This dashboard visualizes the key drivers of customer churn based on historical data.")
    st.markdown("---")

    # --- Create a 2x2 grid for our charts ---
    col1, col2 = st.columns(2)
    st.markdown("---") # Visual separator
    col3, col4 = st.columns(2)

    # --- Chart 1: Churn by Contract ---
    with col1:
        st.subheader('Churn by Contract Type')
        fig, ax = plt.subplots()
        churn_by_contract = df_cleaned.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_contract.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'], ax=ax)
        ax.set_ylabel('Proportion')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(['No Churn', 'Churn'])
        st.pyplot(fig)
        st.markdown("**Insight:** Customers with **Month-to-month contracts** churn at a much higher rate.")

    # --- Chart 2: Churn by Internet Service ---
    with col2:
        st.subheader('Churn by Internet Service')
        fig, ax = plt.subplots()
        churn_by_internet = df_cleaned.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_internet.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'], ax=ax)
        ax.set_ylabel('Proportion')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(['No Churn', 'Churn'])
        st.pyplot(fig)
        st.markdown("**Insight:** Customers with **Fiber optic** internet are more likely to churn.")
    
    # --- Chart 3: Churn by Tenure ---
    with col3:
        st.subheader('Churn by Customer Tenure')
        # Binning tenure for clearer visualization
        bins = [0, 12, 24, 36, 48, 60, 72]
        labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
        df_cleaned['tenure_group'] = pd.cut(df_cleaned['tenure'], bins=bins, labels=labels, right=False)
        
        fig, ax = plt.subplots()
        churn_by_tenure = df_cleaned.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_tenure.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'], ax=ax)
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Tenure (Months)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['No Churn', 'Churn'])
        st.pyplot(fig)
        st.markdown("**Insight:** New customers (**0-12 months**) have the highest churn rate.")

    # --- Chart 4: Churn by Monthly Charges ---
    with col4:
        st.subheader('Churn by Monthly Charges')
        fig, ax = plt.subplots()
        sns.kdeplot(df_cleaned[df_cleaned['Churn']==1]['MonthlyCharges'], label='Churn', color='red', ax=ax)
        sns.kdeplot(df_cleaned[df_cleaned['Churn']==0]['MonthlyCharges'], label='No Churn', color='green', ax=ax)
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_title('Distribution of Monthly Charges')
        ax.legend()
        st.pyplot(fig)
        st.markdown("**Insight:** Customers who churn tend to have higher monthly charges.")

