import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.captioner import ImageCaptioningModel
import os

# Page configuration
st.set_page_config(
    page_title="Image Captioning Demo",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .caption-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        font-size: 1.2rem;
        margin: 20px 0;
        color: #333;
    }
    .info-box {
        padding: 15px;
        background-color: #e8f4f8;
        border-radius: 8px;
        margin: 10px 0;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path):
    """
    Load the trained model (cached for performance).
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        vocab = checkpoint['vocab']
        embed_size = checkpoint['embed_size']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        vocab_size = len(vocab)

        # Initialize model
        model = ImageCaptioningModel(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, vocab
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def preprocess_image(image):
    """
    Preprocess image for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)


def generate_caption(image, model, vocab, max_length=20):
    """
    Generate caption for the image.
    """
    # Preprocess
    image_tensor = preprocess_image(image)

    # Generate caption
    with torch.no_grad():
        caption_indices = model.generate_caption(image_tensor, max_length)

    # Convert indices to words
    caption_words = []
    for idx in caption_indices:
        word = vocab.itos[idx]
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            caption_words.append(word)

    return " ".join(caption_words)


def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Captioning Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an image and let AI describe it!")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value="../artifacts/models/image_captioning_model4.pth",
            help="Path to your trained model file"
        )

        max_caption_length = st.slider(
            "Max Caption Length",
            min_value=10,
            max_value=30,
            value=20,
            help="Maximum number of words in caption"
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")

        if os.path.exists(checkpoint_path):
            st.success("✅ Model loaded successfully!")

            # Load model to show info
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            st.info(f"""
            **Vocabulary Size:** {len(checkpoint['vocab'])}  
            **Embedding Size:** {checkpoint['embed_size']}  
            **Hidden Size:** {checkpoint['hidden_size']}
            """)
        else:
            st.error("❌ Model file not found!")
            st.info("Please train your model first:\n```bash\npython train.py\n```")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image"
        )

        # Sample images option
        st.markdown("---")
        st.markdown("**Or try a sample image:**")

        sample_dir = "test_images"
        if os.path.exists(sample_dir):
            sample_images = [f for f in os.listdir(sample_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if sample_images:
                selected_sample = st.selectbox(
                    "Select sample image",
                    options=["None"] + sample_images
                )

                if selected_sample != "None":
                    uploaded_file = open(os.path.join(sample_dir, selected_sample), 'rb')

    with col2:
        st.subheader("✨ Generated Caption")

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            # Check if model exists
            if not os.path.exists(checkpoint_path):
                st.error("⚠️ Please load a trained model first!")
                st.stop()

            # Load model
            with st.spinner("Loading model..."):
                model, vocab = load_model(checkpoint_path)

            if model is None or vocab is None:
                st.error("Failed to load model. Please check the checkpoint path.")
                st.stop()

            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary", use_container_width=True):
                with st.spinner("Generating caption..."):
                    try:
                        caption = generate_caption(image, model, vocab, max_caption_length)

                        # Display caption in a nice box
                        st.markdown(
                            f'<div class="caption-box">📝 <strong>{caption.capitalize()}</strong></div>',
                            unsafe_allow_html=True
                        )

                        # Additional info
                        st.success("✅ Caption generated successfully!")

                        # Show some stats
                        word_count = len(caption.split())
                        st.markdown(
                            f'<div class="info-box">📊 Caption length: {word_count} words</div>',
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        st.error(f"Error generating caption: {e}")

        else:
            st.info("👆 Upload an image to get started!")

            # Show example
            st.markdown("---")
            st.markdown("**Example captions:**")
            st.markdown("""
            - 🐕 *"a dog running in the park"*
            - 🌊 *"a person surfing on a wave"*
            - 🍕 *"a pizza on a wooden table"*
            - 🏙️ *"a city skyline at night"*
            """)

    # Footer with instructions
    st.markdown("---")
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Quick Start Guide

        1. **Train your model** (if you haven't already):
           ```bash
           python train.py
           ```

        2. **Place test images** in the `test_images/` folder (optional)

        3. **Upload an image** using the file uploader

        4. **Click "Generate Caption"** to see the AI-generated description

        ### Tips for best results:
        - ✅ Use clear, well-lit images
        - ✅ Images similar to training data work best
        - ✅ Try different types of images (people, animals, objects, scenes)
        - ✅ The model works best with common objects and scenes

        ### Troubleshooting:
        - If captions are generic, train for more epochs
        - If you get errors, check that the model file exists
        - Make sure images are JPG or PNG format
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit | Image Captioning with PyTorch"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
