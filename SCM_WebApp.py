import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import plotly.express as px

st.title("Supply Chain (Order Quantity) Prediction")
st.markdown("""
<p style='font-size: 14px;'>
This application allows users to predict the <b>Order Item Quantity</b> for various supply chain orders.
By inputting relevant features such as sales, shipping days, discounts, and more, the model provides 
a predicted quantity along with the confidence level of the prediction. This can help businesses make 
informed decisions and optimize inventory management.
</p>
""", unsafe_allow_html=True)

# Sidebar for file upload with a collapsible section
with st.sidebar.expander("🔄 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding_errors='ignore')
    data = data.head(500)  # Adjustable

    features = ['Sales per customer', 'Days for shipping (real)', 'Days for shipment (scheduled)',
                'Benefit per order', 'Late_delivery_risk', 'Order Item Discount',
                'Order Item Discount Rate', 'Order Item Product Price', 'Order Item Profit Ratio',
                'Sales', 'Order Item Total', 'Order Profit Per Order']

    feature_descriptions = {
        'Sales per customer': 'Total sales made per customer.',
        'Days for shipping (real)': 'Actual shipping days of the purchased product.',
        'Days for shipment (scheduled)': 'Days of scheduled delivery of the purchased product.',
        'Benefit per order': 'Earnings per order placed.',
        'Late_delivery_risk': 'Categorical variable that indicates if shipping is late (1) or not (0).',
        'Order Item Discount': 'Order item discount value.',
        'Order Item Discount Rate': 'Order item discount percentage.',
        'Order Item Product Price': 'Price of products without discount.',
        'Order Item Profit Ratio': 'Order Item Profit Ratio.',
        'Sales': 'Value in sales.',
        'Order Item Total': 'Total amount per order.',
        'Order Profit Per Order': 'Order Profit Per Order.',
        'Customer Segment': 'Types of Customers: Consumer, Corporate, Home Office.',
        'Shipping Mode': 'Shipping modes: Standard Class, First Class, Second Class, Same Day.'
    }

    for feature in features:
        if data[feature].lt(0).any():
            mean_value = data[feature].mean()
            data.loc[data[feature] < 0, feature] = mean_value

    data['order_date'] = pd.to_datetime(data['order date (DateOrders)'])
    data['year'] = data['order_date'].dt.year
    data['month'] = data['order_date'].dt.month
    data['day_of_week'] = data['order_date'].dt.dayofweek


    features += ['Customer Segment', 'Shipping Mode']
    X = data[features]
    X = pd.get_dummies(X) 
    y = data['Order Item Quantity']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # GBC initialization
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    prediction = None
    confidence = None

    # Two-column layout for inputs and results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔮 Make a Prediction")
        with st.sidebar.expander("Make a Prediction", expanded=False):
            new_data = {}
            for feature in features:
                description = feature_descriptions.get(feature, "")
                if feature in data.select_dtypes(include=[np.number]).columns:
                    new_data[feature] = st.number_input(f"Enter {feature}", key=f"{feature}_num_input", value=float(data[feature].median()), help=description)
                else:
                    unique_values = data[feature].unique()
                    new_data[feature] = st.selectbox(f"Select {feature}", unique_values, key=f"{feature}_selectbox", help=description)

            # Button for making predictions with a unique key
            if st.button("Predict", key="predict_button"):
                new_data_df = pd.DataFrame([new_data])
                new_data_df = pd.get_dummies(new_data_df).reindex(columns=X.columns, fill_value=0)
                prediction = model.predict(new_data_df)[0]
                confidence = model.predict_proba(new_data_df)[0].max()

    with col2:
        st.subheader("Model Results")
        if prediction is not None and confidence is not None:
            col2.markdown(f"**Predicted Order Item Quantity:** {prediction}")
            col2.markdown(f"**Confidence of Prediction:** {confidence:.4f}")

    # Dynamic Visualization using Plotly
    st.subheader("Data Summary and Visualization")
    st.write(data.describe())
    fig = px.histogram(data, x='Sales', title='Sales Distribution')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Please upload a CSV file to proceed.")
