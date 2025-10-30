"""
AI Assistant Integration for Equity Ally
Provides contextual analysis, explanations, recommendations, and content improvements
Uses retrieval-augmented generation (RAG) with large language models
"""

import streamlit as st
import os

# Optional import - gracefully handle if openai is not installed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def validate_api_key(api_key):
    """
    Validate that the API key is properly formatted
    
    Args:
        api_key: The OpenAI API key to validate
        
    Returns:
        bool: True if valid format, False otherwise
    """
    if not api_key:
        return False
    return api_key.startswith('sk-') and len(api_key) > 20


def get_completion(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=800, stream=False):
    """
    Get a completion from OpenAI GPT
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model to use (gpt-4o-mini is default for speed/cost, gpt-4o available)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response (returns generator if True)
        
    Returns:
        str or generator: The completion text, or None if error
    """
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package is not installed. Please run: pip install openai")
        return None
    
    try:
        api_key = st.session_state.get('openai_api_key')
        if not validate_api_key(api_key):
            return None
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            # Return generator for streaming
            def generate():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return generate()
        else:
            return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None


def explain_detection(text, prediction, prob_bullying, prob_not_bullying, severity):
    """
    Generate a plain-language explanation of the BERT detection result
    
    Args:
        text: The original text that was analyzed
        prediction: The prediction label ("Concerning Content" or "Content Appears Safe")
        prob_bullying: Probability of concerning content (0-1)
        prob_not_bullying: Probability of safe content (0-1)
        severity: Severity level (Low/Moderate/High Concern)
        
    Returns:
        str: Explanation text or None
    """
    system_prompt = """You are an insightful content safety analyst for Equity Ally. Your role is to help users understand AI detection results by analyzing the actual content and explaining your reasoning in clear, conversational language.

When analyzing text:
- Reference SPECIFIC words, phrases, or patterns you notice in the text
- Explain HOW those elements contribute to the classification
- Consider tone, context, and potential intent
- Be direct about what you observe, but remain empathetic and educational
- Acknowledge when the AI might misinterpret sarcasm, cultural references, or educational content
- Make your explanation feel like a thoughtful human analysis, not a generic template

Keep your explanations concise (2-3 paragraphs) and always ground them in the actual text content provided."""

    user_prompt = f"""Analyze this text and explain why the AI classified it this way:

Text: "{text}"
Classification: {prediction}
Risk Score: {prob_bullying*100:.1f}% concerning, {prob_not_bullying*100:.1f}% safe
Severity: {severity}

Provide a thoughtful explanation that:
1. Identifies the SPECIFIC language patterns, words, or phrases that influenced the classification
2. Explains WHY these elements are concerning (or not), referencing the actual content
3. Notes any important context - could this be sarcasm, a quote, educational discussion, or misinterpreted? What should moderators consider?

Write naturally and reference the actual text - help the user understand what the AI "saw" in their specific content."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return get_completion(messages, temperature=0.7, max_tokens=500)


def get_recommendations(text, prediction, prob_bullying, severity):
    """
    Generate actionable recommendations based on the detection result
    
    Args:
        text: The original text
        prediction: The prediction label
        prob_bullying: Probability of concerning content
        severity: Severity level
        
    Returns:
        str: Recommendations text or None
    """
    system_prompt = """You are an experienced content moderation strategist who understands both AI detection and human judgment. Provide tailored, specific recommendations based on the actual content and context.

Your recommendations should:
- Be SPECIFIC to the text provided, not generic advice
- Reference particular aspects of the content when making suggestions
- Balance safety concerns with fairness and education
- Consider the context, tone, and likely intent of the message
- Differentiate between truly harmful content, misunderstandings, and borderline cases
- Provide clear action items that acknowledge human judgment is essential

Make your advice practical and grounded in the specific situation, not boilerplate responses."""

    user_prompt = f"""A content moderation system flagged this content. Provide specific, actionable recommendations for the moderation team:

Text: "{text}"
AI Classification: {prediction}
Risk Score: {prob_bullying*100:.1f}%
Severity Level: {severity}

Based on THIS specific content, provide 4-5 clear, numbered recommendations that address:
1. Whether human review is needed and why (reference specific aspects of the text)
2. Urgency level and timeline for action
3. Specific next steps tailored to this situation
4. Whether an educational approach or enforcement is more appropriate (explain why for this case)
5. What would warrant escalation based on the content

Ground your recommendations in the actual text - reference specific elements that inform your suggestions."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return get_completion(messages, temperature=0.6, max_tokens=500)


def suggest_improvements(text, prob_bullying):
    """
    Suggest how to rephrase concerning content constructively
    
    Args:
        text: The flagged text
        prob_bullying: Probability of concerning content
        
    Returns:
        str: Improvement suggestions or None
    """
    system_prompt = """You are an empathetic communication coach who helps people express themselves more effectively. Your role is to understand what someone is TRYING to say, then help them say it better.

When suggesting improvements:
- First identify the core message or feeling the person seems to be expressing
- Reference SPECIFIC words or phrases that are problematic and explain WHY
- Offer alternatives that preserve the person's voice and intent while being more constructive
- Show, don't just tell - provide concrete examples using similar language style
- Acknowledge that people deserve to express frustration or disagreement - help them do it constructively
- Be encouraging and educational, never condescending

Think of yourself as a helpful editor who respects the writer's voice while helping them communicate better."""

    user_prompt = f"""This message was flagged as potentially concerning (risk score: {prob_bullying*100:.1f}%):

"{text}"

Help the person communicate more effectively by:
1. Identifying what you think they're TRYING to express (the core message or feeling)
2. Pointing out specific words or phrases that could be problematic and explaining why
3. Offering 2-3 alternative ways to express the same core idea more constructively - keep their voice and intent, just reframe it
4. Sharing 1-2 key communication principles that apply to this specific situation

Be specific, reference the actual text, and make your suggestions practical and respectful."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return get_completion(messages, temperature=0.7, max_tokens=500)


def chat_about_model(user_question, context="", stream=False):
    """
    Answer questions about the Equity Ally model, content safety, or AI in general
    
    Args:
        user_question: The user's question
        context: Optional context about the current conversation
        stream: Whether to stream the response (returns generator if True)
        
    Returns:
        str or generator: Answer or None
    """
    system_prompt = """You are a knowledgeable and approachable AI assistant specializing in Equity Ally and content safety. You can discuss the technology, explain results, and help users understand content moderation.

ABOUT EQUITY ALLY:

Technical Details:
- BERT-base-uncased model (110M parameters, 418MB)
- 12 transformer layers, 768 hidden units, 12 attention heads
- Fine-tuned on 120,000+ samples from 5 toxicity datasets
- Isotonic calibration for optimal probability estimates
- Performance: 91.5% accuracy, 96.8% ROC-AUC, 67.2% F1, 79.3% PR-AUC
- Fast inference: <100ms on CPU, runs completely offline
- Privacy-first: All processing happens on-device

Key Features:
- Monte Carlo Dropout (20-30 passes) for uncertainty quantification
- Optional GPT-4o-mini layer for contextual validation of uncertain cases
- Free, open-source, accessible to everyone
- Works without internet - your data never leaves your device

Your Communication Style:
- Be conversational and engaging, not robotic or overly formal
- When answering about a specific analysis (if context is provided), reference the actual situation
- Adjust technical depth to match the question - be detailed for technical questions, simple for basic ones
- Use examples and analogies when explaining complex concepts
- If the user's question relates to their analyzed text, connect your answer to their specific case
- Admit uncertainty when appropriate - don't make up information

You can discuss: technical architecture, how the model works, performance metrics, content moderation best practices, result interpretation, cyberbullying research, AI/ML concepts, and general questions about online safety."""

    if context:
        user_message = f"Context about the current analysis: {context}\n\nUser's question: {user_question}\n\n(Note: Use the context to make your answer more relevant and specific to their situation)"
    else:
        user_message = user_question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return get_completion(messages, temperature=0.7, max_tokens=600, stream=stream)


def verify_with_gpt(text, bert_prediction, uncertainty):
    """
    Use GPT to verify borderline or uncertain BERT predictions
    This is the Layer 3 of the Triple AI system
    
    Args:
        text: The original text
        bert_prediction: BERT's prediction (0 or 1)
        uncertainty: Uncertainty level from Monte Carlo Dropout
        
    Returns:
        dict: {"verified_prediction": int, "confidence": float, "reasoning": str}
    """
    system_prompt = """You are a content safety verifier with strong contextual understanding. Your job is to validate uncertain AI predictions by analyzing the actual meaning, intent, and context.

When verifying content:
- Look at SPECIFIC words, phrases, and overall message
- Distinguish between: genuine harm, sarcasm, quotes, educational discussion, frustrated venting, and cultural expressions
- Consider: What is the likely INTENT? Who is the likely audience? What is the broader context?
- Recognize that tone, emojis, and phrasing matter significantly
- Be discerning - not everything negative is harmful, and not everything polite is safe

Make your judgment based on whether a reasonable moderator would consider this genuinely concerning or not."""

    user_prompt = f"""The AI model is uncertain about this text. Provide your human-like assessment:

Text: "{text}"
AI Prediction: {"Concerning" if bert_prediction == 1 else "Safe"}
AI Uncertainty: {uncertainty*100:.2f}%

Analyze the text carefully - what is actually being said? What's the tone and likely intent? Then provide:

VERDICT: [concerning/safe]
CONFIDENCE: [0-100]
REASONING: [Explain what you observe in the text that led to your verdict - reference specific elements]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = get_completion(messages, temperature=0.3, max_tokens=300)
    
    if not response:
        return None
    
    # Parse the response
    try:
        lines = response.strip().split('\n')
        verdict = "safe"
        confidence = 50.0
        reasoning = ""
        
        for line in lines:
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().lower()
            elif line.startswith("CONFIDENCE:"):
                confidence = float(line.split(":", 1)[1].strip())
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return {
            "verified_prediction": 1 if "concern" in verdict else 0,
            "confidence": confidence / 100.0,
            "reasoning": reasoning
        }
    except:
        return None


def get_education_content(topic, stream=False):
    """
    Generate educational content about cyberbullying, content safety, or related topics
    
    Args:
        topic: The topic to generate content about
        stream: Whether to stream the response (returns generator if True)
        
    Returns:
        str or generator: Educational content or None
    """
    system_prompt = """You are an engaging educator specializing in digital safety, cyberbullying prevention, and online well-being. Your goal is to inform and empower, not lecture or scare.

Your educational style:
- Start with relatable scenarios or questions that connect to real experiences
- Use concrete examples rather than abstract concepts
- Present research and statistics in digestible, meaningful ways
- Provide actionable advice that people can actually implement
- Balance being informative with being approachable
- Acknowledge complexity - avoid oversimplifying
- End with empowering takeaways, not just warnings

Write like you're talking to someone who genuinely wants to understand and improve, not like you're writing a formal report."""

    user_prompt = f"""Create engaging educational content about: {topic}

Structure your response to include:
1. An opening that connects the topic to real-world experiences or questions
2. Clear explanation of the concept with relevant examples
3. Why this matters - include meaningful statistics or research findings
4. Practical, specific tips that people can actually use
5. Key takeaways that empower and inform

Keep it concise (2-3 paragraphs), conversational, and grounded in evidence. Make someone want to learn more, not feel overwhelmed."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return get_completion(messages, temperature=0.7, max_tokens=600, stream=stream)


