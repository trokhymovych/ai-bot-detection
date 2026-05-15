import re
from typing import Union
from findai.entities import UserProfile, ThreadedConversation

MAX_N_LAST_COMMENTS = 3

def split_into_sentences(text):
    # Pattern looks for punctuation followed by one or more whitespace characters.
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return sentences

def get_direct_comments_string(user: UserProfile) -> str:
    direct_comments_string = ""
    for comment in user.previous_direct_comments:
        direct_comments_string += "\n"
        direct_comments_string += f"ORIGINAL CHANNEL POST: ({comment.social_media_post})\n"
        direct_comments_string += f"DIRECT COMMENT: ({comment.message_text})\n"
        direct_comments_string += "---"
        direct_comments_string += "\n"
    return direct_comments_string

    
def get_conversations_string(user_id: Union[int, str], user_threads: list[ThreadedConversation]) -> str:
    conversations_string = ""
    for conversation in user_threads:
        conversations_string += "---"
        conversations_string += "\n"
        conversations_string += f"Previous post (ID={conversation.id}): ({conversation.initial_post})\n"
        if conversation.previous_comments is not None:
            conversations_string += "\n".join([f"-- message_id = {i.id} from USER {i.user_id} {f'(our target user)' if i.user_id == user_id else ''} answering to {i.answer_to_message_id}:: {i.message_text}" for i in conversation.previous_comments]) + "\n"
        conversations_string += "---"
        conversations_string += "\n"
    return conversations_string


def get_thread_string(user_id: UserProfile, thread: ThreadedConversation) -> str:
    thread_string = ""
    if thread.previous_comments is not None:
        thread_string += "\n".join([f"-- message_id = {i.id} from USER {i.user_id} {f'(our target user)' if i.user_id == user_id else ''} answering to {i.answer_to_message_id}:: {i.message_text}" for i in thread.previous_comments]) + "\n"
    return thread_string


def filter_last_n_messages(thread: ThreadedConversation, n: int=3) -> ThreadedConversation:
    """
    Filter the last 'n' messages from the thread
    Make sure that the message that the last comment is answering to is included.
    """
    # Getting the message that the last comment is answering:
    if thread.previous_comments is None or len(thread.previous_comments) == 0:
        return thread
    last_comment = thread.previous_comments[-1]
    thread_copy = thread.copy()
    if last_comment.answer_to_message_id is None:
        # If the last comment is not answering to any message, just return the last n messages:
        thread_copy.previous_comments = thread_copy.previous_comments[-n:]
    else:
        # Find the message that the last comment is answering to:
        answer_to_message_id = last_comment.answer_to_message_id
        answer_to_message = [m for m in thread.previous_comments if m.id == answer_to_message_id]
        if len(answer_to_message) == 0:
            # If the message is not found, just return the last n messages:
            thread_copy.previous_comments = thread_copy.previous_comments[-n:]
        else:
            # If the message is found, include it in the last n messages:
            answer_to_message = answer_to_message[0]
            # return the message that the last comment is answering before the last message (total n messages):
            last_n_messages = thread_copy.previous_comments[-n:]
            if answer_to_message.id in [last_n.id for last_n in last_n_messages]:
                # If the message is already in the last n messages, just return them:
                thread_copy.previous_comments = last_n_messages
            else:
                # If the message is not in the last n messages, add it to the last n messages:
                thread_copy.previous_comments = last_n_messages[1:-1] + [answer_to_message] + [last_n_messages[-1]]  
    return thread_copy
