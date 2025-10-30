"""
Equity Ally - Learn More & AI Assistant
Educational resources, AI assistant, and content safety guidance
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.shared import page_config, apply_custom_css, load_logo, api_key_sidebar, api_key_compact, show_ai_status, page_navigation, navigation_footer
from utils.openai_helper import get_education_content, chat_about_model

# Page configuration
page_config("Equity Ally - Learn More", "📚", "wide")

# Apply custom CSS
apply_custom_css()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# === HEADER ===
load_logo("assets/equitylogolong.svg", max_width="500px")

st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <h2 style='font-size: 2.5rem; color: #e8eaed; margin-bottom: 1rem;'>📚 Learn More & AI Assistant</h2>
    <p style='font-size: 1.15rem; color: #c1c7d0; max-width: 800px; margin: 0 auto; line-height: 1.7;'>
        Interactive AI assistant, educational resources, and expert guidance<br/>
        for creating safer, more inclusive online communities
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === ENHANCED AI ASSISTANT ===
st.markdown("""
<h2 style='color: #e8eaed; font-size: 2rem; margin-bottom: 1.5rem;'>🤖 AI Assistant</h2>
""", unsafe_allow_html=True)

# Callback function to handle API key changes
def on_api_key_change():
    """Callback when API key is entered"""
    new_key = st.session_state.get('learn_more_api_key_input')
    if new_key and new_key.strip():
        st.session_state.openai_api_key = new_key.strip()

# Inline API key input - always check fresh from session state
api_key = st.session_state.get('openai_api_key')

if not api_key:
    st.markdown("""
    <div class='info-box' style='border-left-color: #f59e0b;'>
        <strong style='font-size: 1.1rem;'>🔑 Add Your API Key to Get Started</strong><br/><br/>
        Add your OpenAI API key below to unlock the AI Assistant. The assistant can help you:
        <ul style='margin-top: 0.5rem;'>
            <li>Understand how Equity Ally works (BERT, Monte Carlo Dropout, etc.)</li>
            <li>Learn about content safety and cyberbullying prevention</li>
            <li>Get guidance on interpreting results and best practices</li>
            <li>Generate custom educational content on any safety topic</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # API key input section - automatically activates when you paste/enter the key
    with st.expander("🔑 Enter API Key", expanded=True):
        st.markdown("""
        <p style='color: #c1c7d0; font-size: 0.95rem; margin-bottom: 0.75rem; line-height: 1.6;'>
            Paste your OpenAI API key below - the AI Assistant will activate automatically.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Paste your OpenAI API key here - it will activate automatically",
                key="learn_more_api_key_input",
                label_visibility="collapsed",
                on_change=on_api_key_change
            )
        
        with col2:
            # Show status indicator
            if st.session_state.get('openai_api_key'):
                st.markdown("""
                <div style='background: #10b981; padding: 0.75rem 1rem; border-radius: 8px; 
                            text-align: center; margin-top: 0rem;'>
                    <strong style='color: #ffffff; font-size: 0.9rem;'>✓ Active</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #1a1d23; padding: 0.75rem 1rem; border-radius: 8px; 
                            text-align: center; margin-top: 0rem; border: 2px solid #2d3139;'>
                    <a href='https://platform.openai.com/api-keys' target='_blank' 
                       style='color: #84A4FC; text-decoration: none; font-size: 0.9rem;'>Get Key</a>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show example questions
    st.markdown("### 💡 Example Questions You Can Ask:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **About the Model:**
        - How does BERT work?
        - What is Monte Carlo Dropout?
        - How accurate is Equity Ally?
        - Explain precision vs. recall
        """)

    with col2:
        st.markdown("""
        **About Content Safety:**
        - What is cyberbullying?
        - How to interpret results?
        - What are best practices?
        - When to escalate to humans?
        """)
    
    with col3:
        st.markdown("""
        **Educational Content:**
        - Create a lesson on digital citizenship
        - How to talk to teens about online safety
        - Strategies for preventing harassment
        - Building positive online communities
        """)

else:
    # AI Assistant is active
    st.markdown("""
    <div style='background: linear-gradient(135deg, #047857 0%, #059669 100%); 
                padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                border: 2px solid #10b981;'>
        <div style='text-align: center;'>
            <strong style='color: #ffffff; font-size: 1.1rem;'>✅ AI Assistant Active</strong>
            <p style='color: #d1fae5; font-size: 0.95rem; margin: 0.5rem 0 0 0;'>
                Ask me anything or choose a mode below
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode selector
    st.markdown("### 🎯 Choose Assistant Mode:")
    
    mode = st.radio(
        "Select what you'd like help with:",
        ["💬 Chat with AI", "📖 Generate Educational Content", "🎓 Quick Questions"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # MODE 1: Chat Interface
    if mode == "💬 Chat with AI":
        st.markdown("### 💬 Interactive Chat")
        
        # Suggested topics selector (replaces fragile buttons)
        st.markdown("**Quick Topics:**")

        if 'chat_input_value' not in st.session_state:
            st.session_state.chat_input_value = ""

        topic = st.selectbox(
            "",
            [
                "— choose a topic —",
                "🧠 Model & Tech",
                "🎲 Monte Carlo",
                "📊 Performance",
                "🛡️ Best Practices",
            ],
            index=0,
            label_visibility="collapsed",
        )

        preset_map = {
            "🧠 Model & Tech": "Tell me about how the BERT model works in Equity Ally and what makes it effective for detecting concerning content.",
            "🎲 Monte Carlo": "Explain Monte Carlo Dropout - what it is, how it works, and why it improves accuracy.",
            "📊 Performance": "What is Equity Ally's accuracy and how does it compare to commercial solutions?",
            "🛡️ Best Practices": "What are the best practices for content moderation and when should I escalate to human review?",
        }

        if topic in preset_map:
            st.session_state.chat_input_value = preset_map[topic]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display chat history with custom styled bubbles
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style='background: #1a1d23; padding: 1rem 1.5rem; border-radius: 12px; 
                            margin-bottom: 1rem; border-left: 4px solid #1463F3;'>
                    <strong style='color: #84A4FC;'>👤 You:</strong><br/>
                    <span style='color: #c1c7d0; line-height: 1.7;'>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: #0d1117; padding: 1rem 1.5rem; border-radius: 12px; 
                            margin-bottom: 1rem; border-left: 4px solid #10b981;'>
                    <strong style='color: #10b981;'>🤖 AI Assistant:</strong><br/>
                    <span style='color: #c1c7d0; line-height: 1.7;'>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # User input with custom styling
        user_question = st.text_area(
            "Your question:",
            value="",
            placeholder="Ask anything about Equity Ally, content safety, BERT, Monte Carlo Dropout, best practices, etc.",
            height=100,
            key="chat_input_field"
        )
        
        # Control buttons
        col_send, col_clear, col_export = st.columns([2, 1, 1])
        
        with col_send:
            send_button = st.button("📤 Send Message", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.chat_input_value = ""
                st.rerun()
        
        with col_export:
            if st.session_state.chat_history and st.button("💾 Export Chat", use_container_width=True):
                # Create export text
                export_text = "Equity Ally AI Assistant Chat Log\n" + "="*50 + "\n\n"
                for msg in st.session_state.chat_history:
                    role = "You" if msg['role'] == 'user' else "AI Assistant"
                    export_text += f"{role}:\n{msg['content']}\n\n" + "-"*50 + "\n\n"
                
                st.download_button(
                    label="📥 Download",
                    data=export_text,
                    file_name="equity_ally_chat.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        # Handle sending message
        if send_button and user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Build context from chat history
            context = ""
            if len(st.session_state.chat_history) > 1:
                context = "Previous conversation:\n"
                for msg in st.session_state.chat_history[-10:-1]:  # Last 9 messages for better context
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    context += f"{role}: {msg['content']}\n"
            
            # Stream AI response
            response_stream = chat_about_model(user_question, context, stream=True)
            
            if response_stream:
                # Stream and collect response
                full_response = st.write_stream(response_stream)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': full_response
                })
                st.rerun()
            else:
                st.error("❌ Failed to get response. Please check your API key.")
        
        # Tips
        if not st.session_state.chat_history:
            st.markdown("""
            <div class='info-box' style='border-left-color: #1463F3; margin-top: 1rem;'>
                <strong>💡 Tips for Great Conversations:</strong><br/><br/>
                • Be specific and detailed in your questions<br/>
                • Ask follow-up questions to dive deeper<br/>
                • Request examples or step-by-step explanations<br/>
                • The AI remembers your conversation context<br/>
                • You can export your chat history for reference
    </div>
    """, unsafe_allow_html=True)
    
    # MODE 2: Educational Content Generator
    elif mode == "📖 Generate Educational Content":
        st.markdown("### 📖 Educational Content Generator")
        
        st.markdown("""
        Generate custom educational materials about cyberbullying, content safety, digital citizenship, 
        and online well-being tailored to your specific needs.
        """)
        
        # Pre-defined templates selector (replaces fragile buttons)
        st.markdown("**📋 Quick Templates:**")

        if 'edu_topic_value' not in st.session_state:
            st.session_state.edu_topic_value = ""

        template = st.selectbox(
            "",
            [
                "— choose a template —",
                "👨‍👩‍👧 Parent Guide",
                "🏫 Teacher Lesson",
                "👥 Community Workshop",
            ],
            index=0,
            label_visibility="collapsed",
        )

        template_map = {
            "👨‍👩‍👧 Parent Guide": "Create a comprehensive guide for parents on how to talk to their teens about cyberbullying and online safety, including warning signs and prevention strategies",
            "🏫 Teacher Lesson": "Create a lesson plan for teachers about digital citizenship and preventing cyberbullying, suitable for middle school students",
            "👥 Community Workshop": "Create content for a community workshop on building safer online spaces and responding to online harassment",
        }

        if template in template_map:
            st.session_state.edu_topic_value = template_map[template]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Custom topic input
        custom_topic = st.text_area(
            "Or describe your custom topic:",
            value=st.session_state.edu_topic_value,
            placeholder="e.g., Create a training module for moderators on handling borderline cases and applying community guidelines fairly",
            height=100,
            key="edu_topic"
        )
        
        # Update the session state value when user types
        if custom_topic != st.session_state.edu_topic_value:
            st.session_state.edu_topic_value = custom_topic
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            generate_button = st.button("📚 Generate Educational Content", type="primary", use_container_width=True)
        
        with col2:
            audience = st.selectbox("Target Audience:", ["General", "Parents", "Educators", "Moderators", "Teens"])
        
        if generate_button and custom_topic:
            full_prompt = f"{custom_topic}\n\nTarget audience: {audience}"
            
            # Stream the educational content
            content_stream = get_education_content(full_prompt, stream=True)
            
            if content_stream:
                st.markdown("---")
                st.markdown("### 📖 Generated Educational Content")
                
                # Stream the content and collect it
                content = st.write_stream(content_stream)
                
                st.markdown("---")
                
                # Download button
                st.download_button(
                    label="💾 Download Content",
                    data=content,
                    file_name="educational_content.txt",
                    mime="text/plain"
                )
    
    # MODE 3: Quick Questions
    else:  # Quick Questions mode
        st.markdown("### 🎓 Quick Questions")
        
        st.markdown("Click any question to get an instant answer:")
        
        # Initialize quick answer tracking
        if 'quick_answer_q' not in st.session_state:
            st.session_state.quick_answer_q = ""
        
        # Organized by category
        st.markdown("#### 🧠 Technical & Model Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("How does BERT work?", use_container_width=True, key="q1"):
                st.session_state.quick_answer_q = "Explain how the BERT model architecture works in Equity Ally, including the transformer layers, attention mechanism, and fine-tuning process."
            
            if st.button("What is Monte Carlo Dropout?", use_container_width=True, key="q2"):
                st.session_state.quick_answer_q = "Explain Monte Carlo Dropout in detail - what it is, how it provides uncertainty estimates, and how it complements the fine-tuned and calibrated BERT model (96.8% accuracy)."
            
            if st.button("How is the model trained?", use_container_width=True, key="q3"):
                st.session_state.quick_answer_q = "Describe the training process for the BERT model including the datasets used, training methodology, and calibration techniques."
        
        with col2:
            if st.button("What makes it accurate?", use_container_width=True, key="q4"):
                st.session_state.quick_answer_q = "Explain why Equity Ally achieves 96.8% accuracy - discuss the two-stage pipeline (fine-tuning on 120K samples + Isotonic calibration)."
            
            if st.button("Precision vs Recall?", use_container_width=True, key="q5"):
                st.session_state.quick_answer_q = "Explain the difference between precision and recall in the context of content moderation, and why Equity Ally prioritizes high recall."
            
            if st.button("How does uncertainty work?", use_container_width=True, key="q6"):
                st.session_state.quick_answer_q = "Explain how uncertainty quantification works with Monte Carlo Dropout and how to interpret the ± values in results."
        
        st.markdown("#### 🛡️ Content Safety & Best Practices")
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("How to interpret results?", use_container_width=True, key="q7"):
                st.session_state.quick_answer_q = "Explain how to interpret detection results including confidence scores, risk levels, and when to trust the model's predictions."
            
            if st.button("When to escalate?", use_container_width=True, key="q8"):
                st.session_state.quick_answer_q = "Explain when content should be escalated to human review and what factors indicate a case needs manual judgment."
            
            if st.button("Handling false positives?", use_container_width=True, key="q9"):
                st.session_state.quick_answer_q = "Explain common causes of false positives (sarcasm, quotes, education) and how to handle them appropriately."
        
        with col4:
            if st.button("Best practices?", use_container_width=True, key="q10"):
                st.session_state.quick_answer_q = "What are the best practices for content moderation using Equity Ally, including the balance of AI and human judgment?"
            
            if st.button("Privacy & compliance?", use_container_width=True, key="q11"):
                st.session_state.quick_answer_q = "Explain how Equity Ally handles privacy, what data is processed, and how it complies with COPPA and FERPA."
            
            if st.button("Comparison to commercial tools?", use_container_width=True, key="q12"):
                st.session_state.quick_answer_q = "Compare Equity Ally to commercial content moderation solutions in terms of accuracy, privacy, and features."
        
        # Display quick answer
        if st.session_state.quick_answer_q:
            question = st.session_state.quick_answer_q
            
            # Stream the response
            response_stream = chat_about_model(question, "", stream=True)
            
            if response_stream:
                st.markdown("---")
                st.markdown("### 🤖 Answer")
                
                # Stream the response and collect it
                response = st.write_stream(response_stream)
                
                st.markdown("---")
                
                col_continue, col_clear = st.columns([1, 1])
                
                # Option to continue in chat
                with col_continue:
                    if st.button("💬 Continue in Chat Mode", use_container_width=True):
                        st.session_state.chat_history = [
                            {'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': response}
                        ]
                        st.session_state.quick_answer_q = ""
                        st.rerun()
                
                with col_clear:
                    if st.button("🗑️ Clear Answer", use_container_width=True):
                        st.session_state.quick_answer_q = ""
                        st.rerun()

st.markdown("---")

# === FAQ SECTION ===
st.markdown("""
<h2 style='color: #e8eaed; font-size: 2rem; margin-bottom: 1.5rem;'>❓ Frequently Asked Questions</h2>
""", unsafe_allow_html=True)

with st.expander("How accurate is Equity Ally?"):
    st.markdown("""
    **Performance by Configuration:**
    
    • **Fine-Tuned + Isotonic Calibration:** 96.8% validation accuracy, 91.5% test accuracy
    
    • **With Optional Monte Carlo Dropout:** Provides uncertainty estimates for borderline cases
    
    • **With AI Assistant:** Additional context and explanations for edge cases
    
    This matches or exceeds commercial content moderation platforms (typically 95-96% accuracy) while 
    maintaining privacy and running entirely on-device.
    """)

with st.expander("Is my data private and secure?"):
    st.markdown("""
    **Yes, Equity Ally is designed with privacy first:**
    
    • **BERT Detection:** Runs entirely on your device, no internet required
    
    • **No Data Storage:** We don't store any analyzed content
    
    • **Your API Key:** Only stored in your browser session, never on our servers
    
    • **OpenAI Calls:** Only when you use AI Assistant features (optional)
    
    • **Compliance:** COPPA, FERPA, and student privacy law compliant
    
    You have complete control over your data and what gets processed.
    """)

with st.expander("Can I use this for my school or organization?"):
    st.markdown("""
    **Absolutely! Equity Ally is perfect for:**
    
    • **Schools & Districts:** Moderate student platforms and forums
    
    • **Youth Organizations:** Create safer online spaces
    
    • **Nonprofits:** Support anti-bullying missions
    
    • **Community Groups:** Protect members from harassment
    
    It's 100% free and open-source. You can deploy it on your own infrastructure, 
    customize it for your needs, and use it without any licensing fees or subscriptions.
    """)

with st.expander("What should I do if the model makes a mistake?"):
    st.markdown("""
    **AI models aren't perfect. Here's what to do:**
    
    **For False Positives (incorrectly flagged as concerning):**
    1. Review the context - sarcasm, quotes, education may be misunderstood
    2. Use AI verification for contextual analysis and a second opinion
    3. Apply human judgment based on your community guidelines
    4. Document patterns to understand model limitations
    
    **For False Negatives (missed harmful content):**
    1. Report it through your moderation process
    2. Consider using Monte Carlo Dropout for better edge case handling
    3. Encourage users to report concerning content
    4. Supplement AI with human review
    
    **Remember:** AI is a tool to assist humans, not replace them.
    """)

with st.expander("How does this compare to commercial solutions?"):
    st.markdown("""
    **Equity Ally vs. Commercial Platforms:**
    
    **Advantages:**
    • **Proven Performance:** 96.8% accuracy (fine-tuned + Isotonic calibration)
    • **Free & Open Source:** No subscriptions or per-user fees
    • **Privacy First:** Runs locally, data stays on your device
    • **Customizable:** Adapt to your specific needs
    • **Transparent:** You can see exactly how it works
    • **Educational:** Built-in guidance and learning resources
    
    **Trade-offs:**
    • Requires basic technical setup (Streamlit, Python)
    • AI assistant features require your own API key (optional)
    • Less polished UI than enterprise products
    • Community support vs. dedicated customer service
    
    For schools, nonprofits, and budget-conscious organizations, Equity Ally 
    offers better performance at no cost.
    """)

st.markdown("---")

# === ADDITIONAL RESOURCES ===
st.markdown("""
<h2 style='color: #e8eaed; font-size: 2rem; margin-bottom: 1.5rem;'>🔗 Additional Resources</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Research & Statistics
    
    • [Pew Research Center](https://www.pewresearch.org/) - Teens & Technology
    
    • [Cyberbullying Research Center](https://cyberbullying.org/)
    
    • [StopBullying.gov](https://www.stopbullying.gov/) - Federal Resources
    
    • [PubMed](https://www.ncbi.nlm.nih.gov/) - Academic Research
    
    • [arXiv](https://arxiv.org/) - AI & ML Research Papers
    """)
    
    st.markdown("""
    ### Support Hotlines
    
    • **Crisis Text Line:** Text HOME to 741741
    
    • **National Suicide Prevention:** 988
    
    • **Cyberbullying Hotline:** 1-800-273-8255
    
    • **SAMHSA Helpline:** 1-800-662-4357
    
    • **Trevor Project (LGBTQ+):** 1-866-488-7386
    """)

with col2:
    st.markdown("""
    ### Educational Organizations
    
    • [Common Sense Media](https://www.commonsensemedia.org/)
    
    • [Internet Safety 101](https://www.internetsafety101.org/)
    
    • [NetSmartz](https://www.netsmartz.org/) - Online Safety Education
    
    • [PACER's National Bullying Prevention Center](https://www.pacer.org/bullying/)
    
    • [Learning for Justice](https://www.tolerance.org/)
    """)
    
    st.markdown("""
    ### Technical Documentation
    
    • [BERT Model Documentation](https://huggingface.co/bert-base-uncased)
    
    • [OpenAI API Documentation](https://platform.openai.com/docs)
    
    • [Streamlit Documentation](https://streamlit.io/)
    
    • [Project Documentation](https://github.com/) (in /docs folder)
    """)

# Page navigation
page_navigation()

# Navigation footer
navigation_footer()

