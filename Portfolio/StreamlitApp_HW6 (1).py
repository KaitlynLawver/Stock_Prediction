import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load



# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features()

MODEL_INFO = {
         "target": "NFLX",
    "endpoint": aws_endpoint,
    "explainer": 'explainer_sentiment.shap',
    "pipeline": 'finalized_sentiment_model.tar.gz',
    "keys": ['ADBE', 'GOOG', 'AMZN', 'sentiment_textblob'],  # ← match your trained model
    "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} 
               for k in ['ADBE', 'GOOG', 'AMZN', 'sentiment_textblob']]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename=MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename, 
        Bucket=bucket, 
        Key= f"{key}/{os.path.basename(filename)}")
        # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    local_path = local_path

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        
    with open(local_path, "rb") as f:
        return load(f)
        #return shap.Explainer.load(f)

# Prediction Logic
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer() 
    )

    try:
        # For regression
        # raw_pred = predictor.predict(input_df)
        # pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # return round(float(pred_val), 4), 200
        # For classification
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
        return mapping.get(int(pred_val), "UNKNOWN"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket, 
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
    
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    
    preprocess_steps = [(name, step) for name, step in best_pipeline.steps 
                        if name not in ('sampler', 'model')]
    preprocessing_pipeline = Pipeline(steps=preprocess_steps)
    
    input_df = input_df[MODEL_INFO["keys"]]
    
    try:
        input_df_transformed = preprocessing_pipeline.transform(input_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    try:
        feature_names = preprocessing_pipeline[-1].get_feature_names_out()
    except Exception:
        feature_names = MODEL_INFO["keys"]

    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)
    
    st.subheader("🔍 Decision Transparency (SHAP)")
    
    # Fix: handle both 2D (regression) and 3D (classification) SHAP outputs
    try:
        if len(shap_values.shape) == 3:
            # Classification - pick the predicted class index
            sv = shap_values[0, :, 0]
        else:
            # Regression - straightforward
            sv = shap_values[0]
        
        shap.plots.waterfall(sv, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        top_feature = pd.Series(
            sv.values, 
            index=sv.feature_names
        ).abs().idxmax()
        st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

    except Exception as e:
        st.error(f"SHAP plot failed: {e}")
        # Fallback: show bar chart of raw SHAP values instead
        try:
            vals = shap_values.values[0] if len(shap_values.shape) == 2 else shap_values.values[0, :, 0]
            shap_series = pd.Series(np.abs(vals), index=feature_names)
            fig, ax = plt.subplots()
            shap_series.sort_values().plot(kind='barh', ax=ax, color='tomato')
            ax.set_title('Feature Importance (SHAP)')
            st.pyplot(fig)
            plt.clf()
        except Exception as e2:
            st.error(f"Fallback plot also failed: {e2}")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    # Prepare data
    # base_df = df_features
    # input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)])
    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])
    
    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df,session, aws_bucket)
    else:
        st.error(res)



