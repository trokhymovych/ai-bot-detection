PERSONA_TEMPLATE = {
    "description": {
        "description": "Brief summary of the persona (2 sentences max). Should define the personas communication style.",
        "possible_output": "Free-text (e.g., 'A russian bot critising the Uganda government, asking questions in case of doubt')"
    },
    "languages": {
        "description": "Languages spoken",
        "possible_output": ["English", "Spanish", "French", "Multilingual"]
    },
    "positive_sentiment_topics": {
        "description": "Topics that are usually discussed in a positive way (should be generalized), max 4 words, 2 generalized topics",
        "possible_output": ["Ukrainian war", "Machine learning: binary classification"]
    },
    "negative_sentiment_topics": {
        "description": "Topics that are usually discussed in a negative way (should be generalized), max 4 words, 2 generalized topics",
        "possible_output": ["NFT Art trading", "Israeli-Palestinian conflict"]
    }
}
