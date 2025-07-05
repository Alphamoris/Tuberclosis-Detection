import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

def display_header():
    st.markdown("""
    <div style='text-align: center;'>
        <h1>Tuberculosis Detection System</h1>
        <p>AI-powered detection of tuberculosis from chest X-ray images</p>
    </div>
    """, unsafe_allow_html=True)
    
def display_image_with_overlay(image, prediction, threshold=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    
    result = "TB Positive" if prediction >= threshold else "Normal"
    color = "red" if prediction >= threshold else "green"
    confidence = max(prediction, 1-prediction)
    
    title = f"{result} ({confidence:.1%} confidence)"
    ax.set_title(title, color=color, fontsize=16)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    
    st.image(buf, use_column_width=True)
    
def create_prediction_gauge(value, label, color):
    fig, ax = plt.subplots(figsize=(4, 0.4))
    
    cmap = plt.cm.RdYlGn_r if label == "TB Probability" else plt.cm.RdYlGn
    norm = plt.Normalize(0, 1)
    
    plt.barh([0], [1], color='lightgrey', height=0.4)
    plt.barh([0], [value], color=color, height=0.4)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"{label}: {value:.1%}", fontweight='bold', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    
    return buf

def display_prediction_results(prediction):
    tb_probability = prediction
    normal_probability = 1 - prediction
    
    result = "TB Detected" if tb_probability >= 0.5 else "Normal"
    result_color = "red" if tb_probability >= 0.5 else "green"
    
    st.markdown(f"<h2 style='color: {result_color};'>{result}</h2>", unsafe_allow_html=True)
    
    tb_gauge = create_prediction_gauge(tb_probability, "TB Probability", "red")
    st.image(tb_gauge)
    
    normal_gauge = create_prediction_gauge(normal_probability, "Normal Probability", "green")
    st.image(normal_gauge)
    
    confidence = max(tb_probability, normal_probability)
    confidence_text = "High" if confidence >= 0.9 else "Medium" if confidence >= 0.7 else "Low"
    confidence_color = "green" if confidence >= 0.9 else "orange" if confidence >= 0.7 else "red"
    
    st.markdown(f"<p>Confidence: <span style='color: {confidence_color}; font-weight: bold;'>{confidence_text} ({confidence:.1%})</span></p>", unsafe_allow_html=True)
    
def display_disclaimer():
    st.warning("""
    **Disclaimer**: This tool is intended for educational purposes only and should not be used for medical diagnosis.
    The AI model provides an analysis based on training data, but the final diagnosis should always be made by qualified healthcare professionals.
    """)

def create_comparison_chart(model_results):
    models = list(model_results.keys())
    tb_probs = [results['tb_probability'] for results in model_results.values()]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, tb_probs, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('TB Probability')
    ax.set_title('Model Comparison')
    
    threshold_line = ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    for bar, prob in zip(bars, tb_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2f}', ha='center', va='bottom')
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    
    return buf

def create_batch_results_table(results):
    df = pd.DataFrame(results)
    
    def color_tb_probability(val):
        color = f'background-color: rgba(255, 0, 0, {val})'
        return color
    
    def color_normal_probability(val):
        color = f'background-color: rgba(0, 255, 0, {val})'
        return color
    
    styled_df = df.style.applymap(color_tb_probability, subset=['TB Probability']) \
                         .applymap(color_normal_probability, subset=['Normal Probability']) \
                         .format({
                             'TB Probability': '{:.1%}',
                             'Normal Probability': '{:.1%}'
                         })
    
    return styled_df

def display_file_upload_area():
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
                
            st.image(img_array, caption="Uploaded X-ray", use_column_width=True)
            return img_array
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    return None

def display_sample_images(sample_paths):
    st.subheader("Sample Images")
    st.write("Don't have an X-ray image? Try one of these samples:")
    
    cols = st.columns(len(sample_paths))
    
    for i, (label, path) in enumerate(sample_paths.items()):
        with cols[i]:
            img = Image.open(path)
            st.image(img, caption=label, width=150)
            if st.button(f"Use {label}", key=f"sample_{i}"):
                return np.array(img)
    
    return None

def get_file_download_link(file_path, link_text="Download Results"):
    with open(file_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path.split("/")[-1]}">{link_text}</a>'
    return href 