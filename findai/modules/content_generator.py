from findai.entities import ThreadedConversation, Persona
from findai.modules.utils import get_conversations_string
from findai.modules.content_generator_reduced import ContentGenerator as _ReducedContentGenerator
from typing import Any, Optional, Union


class ContentGenerator(_ReducedContentGenerator):
    """Extends the reduced generator with few-shot examples from previous conversations."""

    def _main_prompt_template(self, user_id: Union[str, int], user_threads: list[ThreadedConversation], thread: ThreadedConversation, persona: Optional[Persona], to_message_id: Optional[Any]) -> str:
        return f"""
        You are an expert in imitating the user (USER_ID={user_id}) style in social media communication.

        You are provided:
            - [USER_PERSONA] Characteristics of user usual communication style.
            - [PREVIOUS CONVERSATIONS] User previous conversations.
            - [CURRENT CONTEXT] The current conversation thread where you should write a message on behalf of the user (USER_ID={user_id})

        Your task is to imitate the user (USER_ID={user_id}) and generate the most likely text the user would write in the [CURRENT CONTEXT].
        Reproduce from the [PREVIOUS CONVERSATIONS] user's (USER_ID={user_id}) tone characteristics, tendency to jokes, reacting to posts, etc.

        [USER PERSONA]
        {persona.dict()}

        [PREVIOUS CONVERSATIONS]
        {get_conversations_string(user_id, user_threads)}

        """

    def _extra_style_constraints(self, user_id: Union[str, int], user_threads: list[ThreadedConversation]) -> str:
        return f"""
            - Use the same language as the user (USER_ID={user_id}) in the [PREVIOUS CONVERSATIONS].
            - Output must strictly preserve raw formatting style from [PREVIOUS CONVERSATIONS].
        """
