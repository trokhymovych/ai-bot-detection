from findai.entities import ThreadedConversation, Message
from findai.modules.utils import get_thread_string
import os
from openai import OpenAI
from findai.entities import Persona
from typing import Any, Optional, Union
import joblib
import copy


MODEL_NAME = "gpt-4.1-mini-2025-04-14"


class ContentGenerator:
    def __init__(self, client=None, model: str = MODEL_NAME, temperature: float = 0.3, verbose: bool = False, log_path: str = None):
        self.log_path = log_path
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model
        self.verbose = verbose
        self.temperature = temperature

    def _main_prompt_template(self, user_id: Union[str, int], user_threads: list[ThreadedConversation], thread: ThreadedConversation, persona: Optional[Persona], to_message_id: Optional[Any]) -> str:
        return f"""
        You are an expert in imitating the user (USER_ID={user_id}) style in social media communication.

        You are provided:
            - [USER_PERSONA] Characteristics of user usual communication style.
            - [CURRENT CONTEXT] The current conversation thread where you should write a message on behalf of the user (USER_ID={user_id})

        Your task is to imitate the user (USER_ID={user_id}) and generate the most likely text the user would write in the [CURRENT CONTEXT].

        [USER PERSONA]
        {persona.dict()}

        """

    def _extra_style_constraints(self, user_id: Union[str, int], user_threads: list[ThreadedConversation]) -> str:
        """Hook for subclasses to inject additional style constraints."""
        return ""

    def _create_llm_prompt_thread(self, user_id: Union[str, int], user_threads: list[ThreadedConversation], thread: ThreadedConversation, persona: Optional[Persona], to_message_id: Optional[Any], original_length: int = 0) -> str:
        prompt = self._main_prompt_template(user_id, user_threads, thread, persona, to_message_id) + f"""
        [CURRENT CONTEXT]:
        [ORIGINAL POST]
        {thread.initial_post}

        [THREAD]
        {get_thread_string(user_id, thread)}
        """

        if original_length > 0:
            prompt += f"""
            [LENGTH CONSTRAINTS] The response should be strictly around {original_length} words.
            """

        prompt += f"""
            [STYLE CONSTRAINTS]
            - No Unicode punctuation. Do not beautify or autocorrect.
            - You should imitate the user's (USER_ID={user_id}) style as closely as possible.
            {self._extra_style_constraints(user_id, user_threads)}
            """

        if (to_message_id is not None) and (thread.id != to_message_id):
            prompt += f"""
            - Your response should naturally pick up the conversation from the message with ID {to_message_id} within the [CURRENT CONTEXT][THREAD].
            """
        else:
            prompt += f"""
            - Naturally pick up the conversation within the [CURRENT CONTEXT][ORIGINAL POST].
            """

        prompt += """
        [OUTPUT FORMAT]
        You should return only the response text (your message in thread). Nothing else.
        Ensure the text style is consistent with user persona.
        """
        return prompt

    def generate_thread(self, user_id: Union[str, int], user_threads: list[ThreadedConversation], thread: ThreadedConversation, persona: Optional[Persona], the_same_length: bool = True, return_only_prompt: bool = False) -> str:
        thread_for_prompt = copy.deepcopy(thread)
        thread_for_prompt.previous_comments = thread_for_prompt.previous_comments[:-1]
        to_message_id = thread.previous_comments[-1].answer_to_message_id
        original_length = len(thread.previous_comments[-1].message_text.split()) if the_same_length else 0

        prompt = self._create_llm_prompt_thread(user_id, user_threads, thread_for_prompt, persona, to_message_id, original_length)

        if self.verbose:
            print(f"Prompt: {prompt}")

        if return_only_prompt:
            return prompt

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            seed=42,
            temperature=self.temperature,
        )
        result_text = response.choices[0].message.content
        self.log_results(user_id, user_threads, thread_for_prompt, thread.previous_comments[-1], persona, prompt, result_text)
        return result_text

    def log_results(self, user_id: Union[str, int], user_threads: list[ThreadedConversation], thread_for_prompt: ThreadedConversation, true_message: Message, persona: Optional[Persona], prompt: str, generated_text: str) -> None:
        if self.log_path is not None:
            joblib.dump(
                {
                    "user_id": user_id,
                    "user_threads": user_threads,
                    "thread_for_prompt": thread_for_prompt,
                    "persona": persona,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "true_message": true_message,
                },
                self.log_path + f"/{user_id}_{thread_for_prompt.id}.pkl",
            )
