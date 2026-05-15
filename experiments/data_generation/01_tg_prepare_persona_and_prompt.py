from tqdm import tqdm
import numpy as np
import joblib
import copy
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import traceback
import torch
from together import Together

from dotenv import load_dotenv
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
print(f"TOGETHER_API_KEY loaded: {TOGETHER_API_KEY[:8]}...{TOGETHER_API_KEY[-4:] if TOGETHER_API_KEY else 'NOT SET'}")

import sys
sys.path.append("../..")

from findai.modules.user_profiler import PersonaGeneratorTogetherAI
from findai.entities import UserProfile
from findai.modules.content_generator import ContentGenerator
from findai.modules.content_generator_reduced import ContentGenerator as ContentGeneratorReduced
from findai.modules.sample_selector import SampleSelector
from findai.modules.utils import filter_last_n_messages


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except RuntimeError:
        pass


telegram_path_input = "../../data/telegram/02_processed_data/telegram_threads_filtered.joblib"
telegram_path_output = "../../data/telegram/03_prepared_data"

channels = [
    'checkmatenews', 'nikolaosanaximandros', 'dzikizachod',
    'oinformantstarday', 'insiderpaper', 
    'militaernews',
    'venezolamentason', 
    'dnybaoguang', 'guzelolalim',
    'moscowach', 'elpaismexico', 'lanouvellefrance',
    'akhbarefori', 'truexanewsua', 'plnewsy', 'notizieitaliane24'
]

MAX_USERS = 200
MAX_TRAIN_SAMPLES = 500
MAX_TEST_SAMPLES = 20
MIN_TEST_SAMPLES = 5
MIN_TEXT_LENGTH = 1
N_MESSAGES = 5
N_WORKERS = 8

TOGETHER_MODEL = "openai/gpt-oss-20b"
together_client = Together(api_key=TOGETHER_API_KEY)

_check = together_client.chat.completions.create(
    model=TOGETHER_MODEL,
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=3,
)
print(f"API check OK — model={_check.model}, tokens used={_check.usage.total_tokens}")

profile_generation = PersonaGeneratorTogetherAI(client=together_client, model=TOGETHER_MODEL, verbose=False)
content_generation = ContentGenerator(verbose=False)
content_generation_reduced = ContentGeneratorReduced(verbose=False)
example_extractor = SampleSelector()

embedding_lock = threading.Lock()

def build_user_conversations(user_id, selected_conversations):
    all_user_conversations = []
    for conversation in selected_conversations:
        limited_comments = []
        for message in conversation.previous_comments:
            limited_comments.append(message)
            if message.user_id == user_id:
                if len(message.message_text) >= MIN_TEXT_LENGTH and len(conversation.initial_post or "") > 1:
                    new_conv = copy.deepcopy(conversation)
                    new_conv.previous_comments = limited_comments
                    all_user_conversations.append(new_conv)
                break

    all_user_conversations = list({conv.id: conv for conv in all_user_conversations}.values())
    all_user_conversations = sorted(all_user_conversations, key=lambda x: x.previous_comments[0].timestamp)
    return all_user_conversations


def process_user(user_id, selected_conversations):
    all_user_conversations = build_user_conversations(user_id, selected_conversations)

    n_test_samples = np.min([MAX_TEST_SAMPLES, int(len(all_user_conversations) * 0.5)])
    if n_test_samples < MIN_TEST_SAMPLES:
        return None, f"Skipping user {user_id}: only {len(all_user_conversations)} conversations available."

    train_conversations = all_user_conversations[:-n_test_samples]
    if len(train_conversations) > MAX_TRAIN_SAMPLES:
        train_conversations = train_conversations[-MAX_TRAIN_SAMPLES:]
    test_conversations = all_user_conversations[-n_test_samples:]

    user_profile_train = UserProfile(user_id=user_id, previous_conversations=train_conversations)

    persona = profile_generation.generate_persona(user_profile_train, save=False)

    with embedding_lock:
        example_extractor.index_samples_fast(train_conversations)
        most_similar_by_thread = {
            test_thread.id: [i[0] for i in example_extractor.search(test_thread, top_k=5)]
            for test_thread in test_conversations
        }

    records = []
    for test_thread in test_conversations:
        most_similar_threads = most_similar_by_thread[test_thread.id]
        filtered_thread = filter_last_n_messages(test_thread, n=N_MESSAGES)

        full_prompt = content_generation.generate_thread(
            user_id, most_similar_threads, filtered_thread, persona,
            return_only_prompt=True,
        )
        reduced_prompt = content_generation_reduced.generate_thread(
            user_id, most_similar_threads, filtered_thread, persona,
            return_only_prompt=True,
        )

        records.append({
            "user_id": user_id,
            "platform": "telegram",
            "channel": None,
            "persona": persona,
            "thread_id": test_thread.id,
            "prompt": full_prompt,
            "reduced_prompt": reduced_prompt,
            "thread": test_thread,
            "most_similar_threads": most_similar_threads,
            "true_message": test_thread.previous_comments[-1].message_text,
        })

    return records, None


for channel in tqdm(channels):
    print(f"Processing channel: {channel}")
    all_conversations_telegram = joblib.load(telegram_path_input)
    all_conversations = all_conversations_telegram[channel]
    del all_conversations_telegram
    clear_memory()

    user_counter = Counter()
    for conversation in all_conversations:
        for message in conversation.previous_comments:
            user_counter[message.user_id] += 1

    user_ids = [k for k, v in user_counter.items() if v > 15]
    print(f"Channel: {channel}, users with >15 messages: {len(user_ids)} / {len(user_counter)} total")

    if len(user_ids) > MAX_USERS + 100:
        print(f"Limiting to {MAX_USERS + 100} users.")
        np.random.seed(42)
        user_ids = list(np.random.choice(user_ids, size=MAX_USERS + 100, replace=False))

    user_ids_set = set(user_ids)
    selected_conversations = [
        conv for conv in all_conversations
        if any(msg.user_id in user_ids_set for msg in conv.previous_comments)
    ]

    del all_conversations
    clear_memory()

    generation_input = []
    users_processed = 0
    user_ids_to_process = user_ids[:MAX_USERS + 100]

    futures_map = {}
    executor = ThreadPoolExecutor(max_workers=N_WORKERS)
    try:
        for user_id in user_ids_to_process:
            future = executor.submit(process_user, user_id, selected_conversations)
            futures_map[future] = user_id

        for future in tqdm(as_completed(futures_map), total=len(futures_map), desc="users"):
            user_id = futures_map[future]
            try:
                records, skip_msg = future.result()
            except Exception as e:
                print(f"User {user_id} failed: {e}")
                traceback.print_exc()
                continue

            if skip_msg:
                print(skip_msg)
                continue

            for rec in records:
                rec["channel"] = channel
            generation_input.extend(records)
            users_processed += 1

            if users_processed >= MAX_USERS:
                print(f"Reached {MAX_USERS} users limit, stopping.")
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    channel_output_path = os.path.join(telegram_path_output, f"raw_generation_input_{channel}.joblib")
    joblib.dump(generation_input, channel_output_path)
    print(f"Saved {len(generation_input)} records for channel '{channel}'.")
