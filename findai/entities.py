from pydantic import BaseModel
from typing import Optional, Union


class Message(BaseModel):
    id: Optional[str]
    message_text: str
    user_id: Union[int, str]
    answer_to_message_id: Optional[Union[int, str]]
    timestamp: Optional[Union[int, str]] 
    metadata: Optional[dict]

    def __str__(self):
      return f"""Message ID={self.id}, USER_ID={self.user_id}, Answer to {self.answer_to_message_id} MESSAGE_TEXT={self.message_text}"""


class ThreadedConversation(BaseModel): 
    id: Optional[str]
    initial_post: Optional[str]
    previous_comments: Optional[list[Message]]

    def __str__(self):
      line_dilimeter = "\n"
      return f"""ThreadedConversation ID={self.id}: INITIAL POST: {self.initial_post} PREVIOUS_COMMENTS: {line_dilimeter.join([str(i) for i in self.previous_comments])}"""


class Persona(BaseModel):
    description: str
    languages: list[str]
    positive_sentiment_topics: list[str]
    negative_sentiment_topics: list[str]


class UserProfile(BaseModel):
    user_id: Union[int, str]
    previous_conversations: list[ThreadedConversation]
