from sentence_transformers import SentenceTransformer
import numpy as np
import gc
import torch

from findai.entities import ThreadedConversation
from findai.modules.utils import filter_last_n_messages


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except RuntimeError:
        pass
    return "cpu"


def _clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except RuntimeError:
            pass


ENCODER_MODEL = "google/embeddinggemma-300m"
N_COMMENTS_TO_KEEP = 1
N_COMMENTS_TO_KEEP_PROMPT = 3
TEXT_LENGTH_LIMIT = 5000


class SampleSelector:
    def __init__(self, model_name: str = ENCODER_MODEL, verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.device = _get_device()
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        self.samples = []
        self.embeddings = None
        self.batch_size = 512

    def index_samples_fast(self, samples: list[ThreadedConversation]):
        del self.embeddings
        del self.samples
        gc.collect()

        self.samples = samples
        embeddings_list = []

        for i in range(0, len(samples), self.batch_size):
            batch_threads = samples[i:i + self.batch_size]
            batch_texts = [self.convert_thread_to_string(s)[:TEXT_LENGTH_LIMIT] for s in batch_threads]

            with torch.no_grad():
                encoded_batch = self.encoder.encode_document(
                    batch_texts,
                    show_progress_bar=self.verbose,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=32,
                )
            embeddings_list.append(encoded_batch)
            gc.collect()
            _clear_device_cache()

        self.embeddings = np.vstack(embeddings_list)

    def search(self, query: ThreadedConversation, top_k: int = 5):
        query_text = self.convert_thread_to_string(query, mode="test")
        with torch.no_grad():
            query_embedding = self.encoder.encode_query(
                query_text,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        distances = np.dot(self.embeddings, query_embedding.T).flatten()
        top_k_idx = np.argsort(distances)[::-1][:top_k]
        return [(filter_last_n_messages(self.samples[i], N_COMMENTS_TO_KEEP_PROMPT), distances[i]) for i in top_k_idx]


    def convert_thread_to_string(self, thread: ThreadedConversation, mode: str = "train") -> str:
        post_string = f"ORIGINAL POST: {thread.initial_post}\n"
        if mode == "train":
            comments_string = "\n".join([
                f"-- message_id = {i.id} from USER {i.user_id} answering to {i.answer_to_message_id}::: {i.message_text}"
                for i in thread.previous_comments[-N_COMMENTS_TO_KEEP - 1:-1]
            ]) + "\n"
        else:
            comments_string = "\n".join([
                f"-- message_id = {i.id} from USER {i.user_id} answering to {i.answer_to_message_id}::: {i.message_text}"
                for i in thread.previous_comments[-N_COMMENTS_TO_KEEP:]
            ]) + "\n"
        return post_string + comments_string
