from typing import Any
from openai import OpenAI
import os
import json
import numpy as np
from collections import Counter
from findai.entities import UserProfile, Persona, ThreadedConversation
from findai.constants import PERSONA_TEMPLATE
from findai.modules.utils import split_into_sentences
from findai.modules.utils import get_direct_comments_string, get_conversations_string, filter_last_n_messages
import joblib


MODEL_NAME = "gpt-4.1-2025-04-14"
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
N_CONVERSATIONS_TRAIN = 10
N_LAST_COMMENTS = 3

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "gpt-oss:120b"
OLLAMA_CLIENT = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

TOGETHER_MODEL_NAME = "openai/gpt-oss-20b"
try:
    from together import Together
    TOGETHER_CLIENT = None
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    print(f"TOGETHER_API_KEY loaded (internally): {TOGETHER_API_KEY[:8]}...{TOGETHER_API_KEY[-4:] if TOGETHER_API_KEY else 'NOT SET'}")
except ImportError:
    TOGETHER_CLIENT = None


class PersonaGeneratorOpenAI:
    def __init__(self, client: Any=CLIENT, model: str = MODEL_NAME, temperature: float=1, verbose: bool=False, log_path: str=None) -> None:
        self.template = PERSONA_TEMPLATE
        self.template_string = "\n".join(
            [f"'{k}': {v['description']}  # example output (example only, not limited): {v['possible_output']}" for k, v in self.template.items() if not v.get("is_manually_generated", False)])
        self.temperature = temperature
        self.client = client
        self.verbose = verbose
        self.log_path = log_path
        self.model = model

    def create_llm_prompt(self, user: UserProfile, max_threads: int=N_CONVERSATIONS_TRAIN) -> str:
        """Build structured prompt with template constraints"""

        # Select random max_threads threads:
        ids = np.random.choice(len(user.previous_conversations), np.min([max_threads, len(user.previous_conversations)]) , replace=False)
        previous_conversations = [filter_last_n_messages(user.previous_conversations[i], n=N_LAST_COMMENTS) for i in ids]

        prompt = f"""
        You're an expert in analyzing social media conversations and defining the persona of [USER] based on its interactions.
        You will be provided with a set of user interactions in form of conversation under the post, where the user has taken part in ([CONVERSATIONS]). 

        You need to generate a persona JSON for the user based on the provided interactions.
        Pay special attention to users stance to different topics.
        
        The persona JSON should strictly follow the following template:
        {self.template_string}

        Our target [USER] has ID={user.user_id}.

        [CONVERSATIONS]
        {get_conversations_string(user.user_id, previous_conversations)}
        """
        return prompt


    def _get_all_texts(self, user: UserProfile) -> list[str]:
        all_texts = []
        for conversation in user.previous_conversations:
            if conversation.previous_comments is not None:
                all_texts += [i.message_text for i in conversation.previous_comments if i.user_id == user.user_id]
        return all_texts

    
    def generate_persona(self, user: UserProfile, save: bool = True) -> Persona:
        """Generate persona using LLM with template enforcement (use structured output)"""
        prompt = self.create_llm_prompt(user)

        if self.verbose:
            print(f"Prompt: {prompt}")

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=Persona,
            temperature=self.temperature,
            seed=42
        )

        persona_dict = json.loads(completion.choices[0].message.content)
        
        persona = Persona(**persona_dict)

        if self.verbose:
            print(persona)
        if save:
            self.log_results(user, prompt, persona)
        return persona
    
    def log_results(self, user, prompt, result):
        if self.log_path is not None:
            data = {
                "user": user,
                "prompt": prompt,
                "result": result
            }
            joblib.dump(data, self.log_path + f"/{user.user_id}.pkl")
        else:
            pass


class PersonaGeneratorTogetherAI:
    """Persona generator backed by Together.ai via OpenAI-compatible API."""

    def __init__(
        self,
        client: Any = None,
        model: str = TOGETHER_MODEL_NAME,
        temperature: float = 1,
        verbose: bool = False,
        log_path: str = None,
        max_retries: int = 3,
    ) -> None:
        self.template = PERSONA_TEMPLATE
        self.template_string = "\n".join(
            [
                f"'{k}': {v['description']}  # example output (example only, not limited): {v['possible_output']}"
                for k, v in self.template.items()
                if not v.get("is_manually_generated", False)
            ]
        )
        self.temperature = temperature
        self.client = client if client is not None else TOGETHER_CLIENT
        self.verbose = verbose
        self.log_path = log_path
        self.model = model
        self.max_retries = max_retries

    def create_llm_prompt(self, user: UserProfile, max_threads: int = N_CONVERSATIONS_TRAIN) -> str:
        ids = np.random.choice(
            len(user.previous_conversations),
            np.min([max_threads, len(user.previous_conversations)]),
            replace=False,
        )
        previous_conversations = [
            filter_last_n_messages(user.previous_conversations[i], n=N_LAST_COMMENTS) for i in ids
        ]

        prompt = f"""
        You're an expert in analyzing social media conversations and defining the persona of [USER] based on its interactions.
        You will be provided with a set of user interactions in form of conversation under the post, where the user has taken part in ([CONVERSATIONS]).

        You need to generate a persona JSON for the user based on the provided interactions.
        Pay special attention to users stance to different topics.

        The persona JSON should strictly follow the following template:
        {self.template_string}

        Return ONLY a valid JSON object with exactly these keys: "description", "languages", "positive_sentiment_topics", "negative_sentiment_topics".
        Do not include any explanation or markdown — raw JSON only.

        Our target [USER] has ID={user.user_id}.

        [CONVERSATIONS]
        {get_conversations_string(user.user_id, previous_conversations)}
        """
        return prompt

    def generate_persona(self, user: UserProfile, save: bool = True) -> Persona:
        prompt = self.create_llm_prompt(user)

        if self.verbose:
            print(f"Prompt: {prompt}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                persona_dict = json.loads(response.choices[0].message.content)
                persona = Persona(**persona_dict)
                break
            except Exception as e:
                last_error = e
                print(f"[TogetherAI] Attempt {attempt + 1}/{self.max_retries} failed for user {user.user_id}: {e}")
        else:
            raise RuntimeError(
                f"Failed to generate persona for user {user.user_id} after {self.max_retries} attempts"
            ) from last_error

        if self.verbose:
            print(persona)
        if save:
            self.log_results(user, prompt, persona)
        return persona

    def log_results(self, user, prompt, result):
        if self.log_path is not None:
            data = {"user": user, "prompt": prompt, "result": result}
            joblib.dump(data, self.log_path + f"/{user.user_id}.pkl")


class PersonaGeneratorOllama:
    """Persona generator backed by a local Ollama model via OpenAI-compatible API."""

    def __init__(
        self,
        client: Any = OLLAMA_CLIENT,
        model: str = OLLAMA_MODEL_NAME,
        temperature: float = 1,
        verbose: bool = False,
        log_path: str = None,
        max_retries: int = 3,
    ) -> None:
        self.template = PERSONA_TEMPLATE
        self.template_string = "\n".join(
            [
                f"'{k}': {v['description']}  # example output (example only, not limited): {v['possible_output']}"
                for k, v in self.template.items()
                if not v.get("is_manually_generated", False)
            ]
        )
        self.temperature = temperature
        self.client = client
        self.verbose = verbose
        self.log_path = log_path
        self.model = model
        self.max_retries = max_retries

    def create_llm_prompt(self, user: UserProfile, max_threads: int = N_CONVERSATIONS_TRAIN) -> str:
        ids = np.random.choice(
            len(user.previous_conversations),
            np.min([max_threads, len(user.previous_conversations)]),
            replace=False,
        )
        previous_conversations = [
            filter_last_n_messages(user.previous_conversations[i], n=N_LAST_COMMENTS) for i in ids
        ]

        prompt = f"""
        You're an expert in analyzing social media conversations and defining the persona of [USER] based on its interactions.
        You will be provided with a set of user interactions in form of conversation under the post, where the user has taken part in ([CONVERSATIONS]).

        You need to generate a persona JSON for the user based on the provided interactions.
        Pay special attention to users stance to different topics.

        The persona JSON should strictly follow the following template:
        {self.template_string}

        Return ONLY a valid JSON object with exactly these keys: "description", "languages", "positive_sentiment_topics", "negative_sentiment_topics".
        Do not include any explanation or markdown — raw JSON only.

        Our target [USER] has ID={user.user_id}.

        [CONVERSATIONS]
        {get_conversations_string(user.user_id, previous_conversations)}
        """
        return prompt

    def generate_persona(self, user: UserProfile, save: bool = True) -> Persona:
        prompt = self.create_llm_prompt(user)

        if self.verbose:
            print(f"Prompt: {prompt}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    seed=42,
                )
                persona_dict = json.loads(response.choices[0].message.content)
                persona = Persona(**persona_dict)
                break
            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"Attempt {attempt + 1} failed: {e}")
        else:
            raise RuntimeError(
                f"Failed to generate persona for user {user.user_id} after {self.max_retries} attempts"
            ) from last_error

        if self.verbose:
            print(persona)
        if save:
            self.log_results(user, prompt, persona)
        return persona

    def log_results(self, user, prompt, result):
        if self.log_path is not None:
            data = {"user": user, "prompt": prompt, "result": result}
            joblib.dump(data, self.log_path + f"/{user.user_id}.pkl")

