from collections import Counter

from docarray.typing import TorchTensor, TorchEmbedding, ImageUrl
from typing import Optional
from docarray.documents import ImageDoc
from docarray import BaseDoc, DocVec, DocList
from docarray.typing import ID


class MultiModal(BaseDoc):
    embedding: TorchTensor
    path: ImageUrl
    label: int
    label_description: str
    zeroshot_label: int
    zeroshot_description: str
    task: int
    task_description: str
    id_code: str
    metadata: dict
