import streamlit as st
from langchain.prompts import PromptTemplate
import os
from typing import Optional, Tuple
import torch

# Import transformers components
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import transformers
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

class LocalLLMGenerator:
    def __init__(self):
        self.pipeline = None
        self.current_model = None
        
        # Comprehensive model catalog organized by categories
        self.available_models = {
            # Ultra Lightweight Models (< 500M parameters)
            "SmolLM2 135M Instruct": {
                "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "size": "135M",
                "type": "Ultra lightweight",
                "category": "Ultra Light",
                "description": "Fastest, minimal memory usage"
            },
            "SmolLM2 360M Instruct": {
                "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct", 
                "size": "360M",
                "type": "Ultra lightweight",
                "category": "Ultra Light",
                "description": "Very fast, good for simple tasks"
            },
            "DistilGPT-2": {
                "model_id": "distilgpt2",
                "size": "82M",
                "type": "Distilled model",
                "category": "Ultra Light",
                "description": "Classic, reliable, very fast"
            },
            "GPT-2 Small": {
                "model_id": "gpt2",
                "size": "124M",
                "type": "Classic baseline",
                "category": "Ultra Light", 
                "description": "Original GPT-2, good baseline"
            },
            
            # Lightweight Models (500M - 2B parameters)
            "TinyLlama 1.1B Chat": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "size": "1.1B",
                "type": "Chat optimized",
                "category": "Lightweight",
                "description": "Excellent chat capabilities, fast"
            },
            "SmolLM2 1.7B Instruct": {
                "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "size": "1.7B", 
                "type": "Instruction following",
                "category": "Lightweight",
                "description": "Good balance of speed and quality"
            },
            "Phi-2": {
                "model_id": "microsoft/phi-2",
                "size": "2.7B",
                "type": "Microsoft research",
                "category": "Lightweight",
                "description": "Strong reasoning, good for coding"
            },
            "Stable LM Zephyr 3B": {
                "model_id": "stabilityai/stablelm-zephyr-3b",
                "size": "3B",
                "type": "Chat optimized",
                "category": "Lightweight", 
                "description": "Stable AI's chat model"
            },
            
            # Medium Models (3B - 8B parameters)
            "Phi-3 Mini 4K": {
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "size": "3.8B",
                "type": "Microsoft Phi-3",
                "category": "Medium",
                "description": "Latest Microsoft model, great performance"
            },
            "Gemma 2B": {
                "model_id": "google/gemma-2b",
                "size": "2B",
                "type": "Google Gemma",
                "category": "Medium",
                "description": "Google's lightweight model"
            },
            "Gemma 7B": {
                "model_id": "google/gemma-7b",
                "size": "7B", 
                "type": "Google Gemma",
                "category": "Medium",
                "description": "High quality, good reasoning"
            },
            "Qwen2.5 3B Instruct": {
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "size": "3B",
                "type": "Alibaba Qwen",
                "category": "Medium",
                "description": "Excellent multilingual support"
            },
            "Qwen2.5 7B Instruct": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct", 
                "size": "7B",
                "type": "Alibaba Qwen",
                "category": "Medium",
                "description": "High performance, multilingual"
            },
            "Llama 3.2 3B Instruct": {
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                "size": "3B",
                "type": "Meta Llama",
                "category": "Medium",
                "description": "Latest Llama, excellent quality",
                "gated": True,
                "auth_required": "Requires HuggingFace authentication - see sidebar"
            },
            
            # Large Models (8B+ parameters - May require significant resources)
            "Llama 3.1 8B Instruct": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "size": "8B",
                "type": "Meta Llama",
                "category": "Large",
                "description": "High quality, requires more memory",
                "gated": True,
                "auth_required": "Requires HuggingFace authentication - see sidebar"
            },
            "Mistral 7B Instruct": {
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "size": "7B",
                "type": "Mistral AI",
                "category": "Large",
                "description": "Excellent instruction following"
            },
            "Gemma 2 9B": {
                "model_id": "google/gemma-2-9b",
                "size": "9B",
                "type": "Google Gemma 2",
                "category": "Large",
                "description": "Latest Gemma, high performance"
            },
            
            # Specialized Models
            "CodeLlama 7B Python": {
                "model_id": "codellama/CodeLlama-7b-Python-hf",
                "size": "7B",
                "type": "Code generation",
                "category": "Specialized",
                "description": "Optimized for Python coding"
            },
            "CodeT5+ 220M": {
                "model_id": "Salesforce/codet5p-220m",
                "size": "220M", 
                "type": "Code generation",
                "category": "Specialized",
                "description": "Lightweight coding model"
            },
            "Falcon 1B": {
                "model_id": "tiiuae/falcon-rw-1b",
                "size": "1B",
                "type": "Technology Innovation Institute",
                "category": "Specialized",
                "description": "Efficient, good general performance"
            }
        }
        
    def get_models_by_category(self):
        """Organize models by category for better UI"""
        categories = {}
        for model_name, info in self.available_models.items():
            category = info.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(model_name)
        return categories
        
    def check_system_requirements(self) -> Tuple[bool, str]:
        """Check if system can run models"""
        if not HF_TRANSFORMERS_AVAILABLE:
            return False, "❌ Transformers library not installed. Run: pip install transformers torch"
            
        # Check available memory (rough estimate)
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory >= 8:
                    return True, f"✅ GPU available with {gpu_memory:.1f}GB VRAM - Can run large models"
                elif gpu_memory >= 4:
                    return True, f"✅ GPU available with {gpu_memory:.1f}GB VRAM - Recommended: medium models"
                else:
                    return True, f"✅ GPU available with {gpu_memory:.1f}GB VRAM - Recommended: lightweight models"
            else:
                return True, "✅ CPU mode available (slower but works) - Recommended: ultra light models"
        except:
            return True, "✅ System ready for CPU inference - Recommended: ultra light models"
    
    def estimate_memory_usage(self, model_name: str) -> str:
        """Estimate memory usage for a model"""
        if model_name not in self.available_models:
            return "Unknown"
            
        size = self.available_models[model_name]["size"]
        
        # Rough memory estimation (parameters * 2 bytes for float16 + overhead)
        if "M" in size:
            params = float(size.replace("M", ""))
            memory_gb = (params * 2) / 1000  # Convert MB to GB roughly
        elif "B" in size:
            params = float(size.replace("B", ""))
            memory_gb = params * 2  # GB for float16
        else:
            return "Unknown"
            
        if memory_gb < 1:
            return f"~{int(memory_gb * 1000)}MB RAM"
        else:
            return f"~{memory_gb:.1f}GB RAM"
    
    def load_model(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        """Download and load model locally with better error handling"""
        try:
            if model_name not in self.available_models:
                return False, f"Model {model_name} not found"
                
            model_info = self.available_models[model_name]
            model_id = model_info["model_id"]
            
            # Check if model is gated
            is_gated = model_info.get("gated", False)
            if is_gated:
                return False, "❌ This model requires HuggingFace authentication. Try Qwen2.5 3B or TinyLlama 1.1B instead."
            
            if progress_callback:
                progress_callback(f"Loading {model_name} ({model_info['size']})...")
            
            # Determine device and settings
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            
            # Adjust settings based on model size and device
            torch_dtype = torch.float16 if device == 0 else torch.float32
            
            if progress_callback:
                progress_callback(f"Downloading model to {device_name}...")
            
            # Create text generation pipeline with optimized settings
            self.pipeline = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                device=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # Common pad token ID
            )
            
            # Set pad_token_id if not set
            if self.pipeline.tokenizer.pad_token_id is None:
                self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            
            self.current_model = model_name
            
            return True, f"✅ {model_name} loaded successfully on {device_name}"
            
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda out of memory" in error_msg.lower():
                return False, "❌ Out of memory. Try a smaller model or use CPU mode."
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                return False, "❌ Network error. Check your internet connection and try a different model."
            elif "not found" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                return False, f"❌ Model access denied. This model may require authentication. Try TinyLlama 1.1B or Qwen2.5 3B instead."
            elif "gated" in error_msg.lower() or "private" in error_msg.lower():
                return False, f"❌ This model requires HuggingFace authentication. Try Qwen2.5 3B or TinyLlama 1.1B instead."
            else:
                return False, f"❌ Error loading model: {error_msg[:100]}..."
    
    def generate_text(self, prompt: str, max_length: int = 300, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text using loaded model with improved parameters"""
        try:
            if not self.pipeline:
                return "❌ No model loaded. Please load a model first."
            
            # Generate text with better parameters
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.pad_token_id or self.pipeline.tokenizer.eos_token_id,
                num_return_sequences=1,
                truncation=True,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Remove the original prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Clean up common artifacts
            generated_text = generated_text.replace("<|endoftext|>", "").strip()
            
            return generated_text if generated_text else "Generated text was empty. Try adjusting parameters."
            
        except Exception as e:
            return f"❌ Generation error: {str(e)[:100]}..."
    
    def get_model_info(self) -> dict:
        """Get information about currently loaded model"""
        if self.current_model:
            return {
                "name": self.current_model,
                "info": self.available_models[self.current_model],
                "loaded": True
            }
        return {"loaded": False}

def create_blog_prompt(topic: str, word_count: str, style: str) -> str:
    """Create an effective prompt for blog generation"""
    
    style_instructions = {
        "Informative": "Write in an educational and factual tone with clear explanations",
        "Casual": "Write in a friendly and conversational tone as if talking to a friend", 
        "Technical": "Write in a detailed and professional tone with technical accuracy",
        "Creative": "Write in an engaging and imaginative tone with vivid descriptions",
        "Academic": "Write in a scholarly tone with structured arguments",
        "Journalistic": "Write in a news-style tone with facts and quotes"
    }
    
    style_desc = style_instructions.get(style, "informative tone")
    
    prompt = f"""Write a {word_count}-word blog post about {topic}.

{style_desc}.

Topic: {topic}

Blog Post:
"""
    return prompt

def main():
    st.set_page_config(
        page_title="Local LLM Blog Generator Pro",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Local LLM Blog Generator Pro")
    st.markdown("**Run 40+ open-source language models locally - no API needed!**")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = LocalLLMGenerator()
    
    generator = st.session_state.generator
    
    # Sidebar for model management
    with st.sidebar:
        st.header("🤖 Model Management")
        
        # System check
        is_ready, system_msg = generator.check_system_requirements()
        if is_ready:
            st.success(system_msg)
        else:
            st.error(system_msg)
            st.stop()
        
        # Model selection by category
        st.subheader("📥 Choose Your Model")
        
        categories = generator.get_models_by_category()
        
        # Category selector
        selected_category = st.selectbox(
            "Model Category:",
            options=list(categories.keys()),
            index=0,
            help="Choose based on your hardware capabilities"
        )
        
        # Model selector within category
        models_in_category = categories[selected_category]
        selected_model = st.selectbox(
            "Model:",
            options=models_in_category,
            index=0
        )
        
        # Show detailed model info
        model_info = generator.available_models[selected_model]
        memory_usage = generator.estimate_memory_usage(selected_model)
        
        # Check if model requires authentication
        is_gated = model_info.get("gated", False)
        
        if is_gated:
            st.warning(f"""
            ⚠️ **Authentication Required**  
            This model requires HuggingFace authentication.  
            **Alternative:** Try Qwen2.5 3B or TinyLlama 1.1B for similar quality without authentication.
            """)
        
        st.info(f"""
        **Size:** {model_info['size']} parameters  
        **Type:** {model_info['type']}  
        **Memory:** {memory_usage}  
        **Description:** {model_info['description']}
        """)
        
        if is_gated:
            st.info("🔐 **Need this model?** Get HuggingFace token at: https://huggingface.co/settings/tokens")
        
        # Load model button
        col1, col2 = st.columns(2)
        with col1:
            load_button = st.button("📥 Load Model", type="primary", use_container_width=True)
        with col2:
            if st.session_state.generator.current_model:
                clear_button = st.button("🗑️ Clear", use_container_width=True)
                if clear_button:
                    st.session_state.generator.pipeline = None
                    st.session_state.generator.current_model = None
                    st.rerun()
        
        if load_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(message):
                status_text.text(message)
                progress_bar.progress(50)
            
            with st.spinner(f"Loading {selected_model}..."):
                success, message = generator.load_model(selected_model, progress_callback)
                
                progress_bar.progress(100)
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
                    
        # Current model status
        model_status = generator.get_model_info()
        if model_status["loaded"]:
            st.success(f"✅ Loaded: {model_status['name']}")
        else:
            st.warning("⚠️ No model loaded")
            
        st.markdown("---")
        st.markdown("### 💡 Model Recommendations")
        st.markdown("""
        **🏃‍♂️ First time / CPU only:** DistilGPT-2, SmolLM2 135M  
        **⚡ Best speed/quality:** TinyLlama 1.1B, Phi-2  
        **🎯 Best quality (no auth):** Qwen2.5 3B, Gemma 2B  
        **💻 For coding:** CodeT5+ 220M, CodeLlama 7B  
        **🌍 Multilingual:** Qwen2.5 series  
        **⚠️ Gated models:** Llama models need HuggingFace auth
        """)
    
    # Main interface
    if not generator.get_model_info()["loaded"]:
        st.warning("⚠️ Please load a model from the sidebar first")
        st.markdown("""
        ### 🎯 Quick Start Guide:
        1. **Choose a category** based on your hardware (Ultra Light for older computers)
        2. **Select a model** from the dropdown
        3. **Click "Load Model"** and wait for download (first time only)
        4. **Start generating!** Models run completely offline once loaded
        """)
        st.stop()
    
    st.subheader("📝 Blog Generation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "📋 Blog Topic:",
            value="The Benefits of Artificial Intelligence in Healthcare",
            help="What should your blog post be about?"
        )
        
    with col2:
        word_count = st.selectbox(
            "📊 Length:",
            options=["100", "200", "300", "500", "800"],
            index=2
        )
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        style = st.selectbox(
            "🎨 Writing Style:",
            options=["Casual", "Informative", "Technical", "Creative", "Academic", "Journalistic"],
            index=0
        )
        
    with col4:
        temperature = st.slider(
            "🌡️ Creativity:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative but less focused"
        )
        
    with col5:
        top_p = st.slider(
            "🎯 Focus:",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Lower = more focused responses"
        )
    
    # Generation section
    col_gen1, col_gen2 = st.columns([3, 1])
    
    with col_gen1:
        generate_button = st.button("🚀 Generate Blog Post", type="primary", use_container_width=True)
    
    with col_gen2:
        if st.button("🎲 Random Topic", use_container_width=True):
            topics = [
                "The Future of Remote Work",
                "Sustainable Living Tips",
                "Digital Privacy in 2025", 
                "The Rise of Electric Vehicles",
                "Mental Health in the Digital Age",
                "Space Exploration Breakthroughs",
                "The Impact of Social Media",
                "Renewable Energy Solutions"
            ]
            import random
            st.session_state.random_topic = random.choice(topics)
            st.rerun()
    
    # Use random topic if generated
    if hasattr(st.session_state, 'random_topic'):
        topic = st.session_state.random_topic
        delattr(st.session_state, 'random_topic')
    
    if generate_button:
        if not topic.strip():
            st.error("Please enter a blog topic")
            st.stop()
            
        # Create prompt
        prompt = create_blog_prompt(topic, word_count, style)
        
        # Calculate max length
        try:
            target_words = int(word_count)
            max_tokens = len(prompt.split()) + int(target_words * 1.3)
        except:
            max_tokens = 300
            
        st.markdown("## 📖 Generated Blog Post")
        
        with st.spinner("🤔 Thinking and writing..."):
            result = generator.generate_text(
                prompt=prompt,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
        if result.startswith("❌"):
            st.error(result)
        else:
            st.markdown("### 📄 Your Blog Post")
            st.markdown(result)
            
            # Statistics
            word_count_actual = len(result.split())
            char_count = len(result)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("📊 Words", word_count_actual)
            with col_stat2:
                st.metric("📝 Characters", char_count)
            with col_stat3:
                st.metric("🤖 Model", model_status['name'].split()[0])
            
            # Action buttons
            col_action1, col_action2, col_action3 = st.columns(3)
            
            with col_action1:
                st.download_button(
                    "💾 Download (.txt)",
                    data=result,
                    file_name=f"blog_{topic.replace(' ', '_').lower()}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            with col_action2:
                # Create markdown version
                markdown_content = f"# {topic}\n\n{result}"
                st.download_button(
                    "📄 Download (.md)",
                    data=markdown_content,
                    file_name=f"blog_{topic.replace(' ', '_').lower()}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
            with col_action3:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.rerun()

if __name__ == "__main__":
    main()