import numpy as np
import torch
import clip
from collections import Counter
from docarray import DocList

from ape.data_models import MultiModal


def get_label_tokens(embeder, labels_prompts):
    print(labels_prompts)
    label_tokens = clip.tokenize(labels_prompts)
    with torch.no_grad():
        text_embs = embeder.encode_text(label_tokens)
    return text_embs


def stratified_sampling(docs: DocList[MultiModal], zeroshot_labels, n_neighbors=10):
    sampled_docs = []

    for i in np.unique(zeroshot_labels):
        class_docs = [doc for doc, label in zip(docs, zeroshot_labels) if label == i]

        if len(class_docs) > n_neighbors:
            sampled_docs += np.random.choice(class_docs, n_neighbors, replace=False).tolist()
        else:
            sampled_docs += class_docs

    return DocList[MultiModal](sampled_docs)


def zeroshot_clustering(docs: DocList[MultiModal], text_embs, labels_prompts):
    image_embs = torch.stack([doc.embedding for doc in docs])
    image_embs /= image_embs.norm(dim=-1, keepdim=True)

    text_emb = text_embs.clone()
    text_emb /= text_emb.norm(dim=-1, keepdim=True)

    print(f"image_embs: {image_embs.shape}")
    print(f"text_emb: {text_emb.shape}")

    similarity = (100.0 * image_embs @ text_emb.T).softmax(dim=-1)
    zeroshot_labels = torch.argmax(similarity, dim=1).numpy()

    n_clusters_ = len(set(zeroshot_labels))

    print("Estimated number of clusters: %d" % n_clusters_)
    print(f"Counter zero-shot labels = {Counter(zeroshot_labels)}")

    new_docs = DocList[MultiModal](docs)
    new_docs.zeroshot_label = zeroshot_labels
    new_docs.zeroshot_description = [labels_prompts[i] for i in zeroshot_labels]

    return new_docs, zeroshot_labels, n_clusters_


def get_zeroshot_sampling(docs: DocList[MultiModal], text_embs, labels_prompts, n_neighbors=10):
    clustered_docs, zeroshot_labels, n_clusters = zeroshot_clustering(docs, text_embs, labels_prompts)
    sampled_docs = stratified_sampling(clustered_docs, zeroshot_labels, n_neighbors)

    print(f"# Sampled docs ({round((len(sampled_docs) / len(clustered_docs)) * 100, 2)}%):", len(sampled_docs))
    print()

    return clustered_docs, sampled_docs
