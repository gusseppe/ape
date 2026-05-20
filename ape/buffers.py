import numpy as np
import torch
from torch.utils.data import TensorDataset

from avalanche.training.storage_policy import ParametricBuffer, _ParametricSingleBuffer
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import SupervisedPlugin
from avalanche.models import FeatureExtractorBackbone
from avalanche.benchmarks.utils import AvalancheDataset
from docarray import DocList

from ape.data_models import MultiModal
from ape.clip_utils import get_zeroshot_sampling


class RandomExemplarsBuffer(ParametricBuffer):
    """Replay buffer that selects exemplars at random, using model-predicted labels."""

    def __init__(self, max_size, n_neighbors=10, seed=42, groupby=None, selection_strategy=None):
        super().__init__(max_size, groupby, selection_strategy)
        self.n_neighbors = n_neighbors
        self.seed = seed
        print(">>>>RandomExemplarsBuffer2")

    def update(self, strategy, **kwargs):
        dataset_len = len(strategy.experience.dataset)
        if self.n_neighbors > dataset_len:
            raise ValueError("n_neighbors cannot be greater than the size of the dataset")

        np.random.seed(self.seed)
        random_indices = np.random.choice(dataset_len, self.n_neighbors, replace=False)

        x_random = [strategy.experience.dataset[i][0] for i in random_indices]
        t_random = [0 for _ in random_indices]

        x_random_pt = torch.stack(x_random).to(strategy.device)
        t_random_pt = torch.tensor(t_random).to(strategy.device)

        strategy.model.eval()
        with torch.no_grad():
            y_random_pt = strategy.model(x_random_pt, t_random_pt)
            y_random_pred = torch.argmax(y_random_pt, dim=1)

        y_random_pt = y_random_pred.cpu()
        t_random_pt = torch.from_numpy(np.asarray(t_random))

        new_data = TensorDataset(x_random_pt.cpu(), y_random_pt, t_random_pt)
        new_data = AvalancheDataset(new_data)
        new_data.targets = y_random_pt.tolist()

        print("Update x_random: ", len(x_random))
        print("Update y_random (predicted): ", len(y_random_pred))
        print("Update new_data: ", len(new_data))
        print("Update new_data[0]: ", len(new_data[0]))

        new_groups = self._make_groups(strategy, new_data)
        self.seen_groups.update(new_groups.keys())

        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {group_id: ll for group_id, ll in zip(self.seen_groups, lens)}

        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            if group_id in self.buffer_groups:
                old_buffer_g = self.buffer_groups[group_id]
                old_buffer_g.update_from_dataset(strategy, new_data_g)
                old_buffer_g.resize(strategy, ll)
            else:
                new_buffer = _ParametricSingleBuffer(ll, self.selection_strategy)
                new_buffer.update_from_dataset(strategy, new_data_g)
                self.buffer_groups[group_id] = new_buffer

        for group_id in self.buffer_groups:
            self.buffer_groups[group_id].resize(strategy, group_to_len[group_id])


class ZeroshotExemplarsBuffer(ParametricBuffer):
    """Replay buffer that selects exemplars via zero-shot CLIP clustering (APE-guided)."""

    def __init__(self, max_size, text_embs, labels_prompts, n_neighbors=10,
                 groupby=None, selection_strategy=None):
        super().__init__(max_size, groupby, selection_strategy)
        self.n_neighbors = n_neighbors
        self.text_embs = text_embs
        self.labels_prompts = labels_prompts
        print(">>>>ZeroshotExemplarsBuffer")

    def update(self, strategy, **kwargs):
        data_list = [
            MultiModal(
                embedding=data[0], path="", label=data[1], label_description="",
                zeroshot_label=-1, zeroshot_description="",
                task=data[2], task_description="",
                id_code="", metadata={}
            )
            for data in strategy.experience.dataset
        ]
        multimodal_data = DocList[MultiModal](data_list)
        print(type(multimodal_data))

        clustered_docs, sampled_docs = get_zeroshot_sampling(
            multimodal_data, self.text_embs, self.labels_prompts, n_neighbors=self.n_neighbors
        )

        x_knn = sampled_docs.embedding
        y_knn = sampled_docs.zeroshot_label
        t_knn = [0 for _ in y_knn]

        x_knn_pt = torch.stack(x_knn)
        y_knn_pt = torch.from_numpy(np.asarray(y_knn))
        t_knn_pt = torch.from_numpy(np.asarray(t_knn))

        new_data = TensorDataset(x_knn_pt, y_knn_pt, t_knn_pt)
        new_data = AvalancheDataset(new_data)
        new_data.targets = y_knn_pt.tolist()

        print("Update x_knn: ", len(x_knn))
        print("Update y_knn: ", len(y_knn))
        print("Update new_data: ", len(new_data))
        print("Update new_data[0] before: ", len(new_data[0]))

        new_groups = self._make_groups(strategy, new_data)
        self.seen_groups.update(new_groups.keys())

        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {group_id: ll for group_id, ll in zip(self.seen_groups, lens)}

        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            if group_id in self.buffer_groups:
                old_buffer_g = self.buffer_groups[group_id]
                old_buffer_g.update_from_dataset(strategy, new_data_g)
                old_buffer_g.resize(strategy, ll)
            else:
                new_buffer = _ParametricSingleBuffer(ll, self.selection_strategy)
                new_buffer.update_from_dataset(strategy, new_data_g)
                self.buffer_groups[group_id] = new_buffer

        for group_id in self.buffer_groups:
            self.buffer_groups[group_id].resize(strategy, group_to_len[group_id])

        print("Update new_data[0] after: ", len(new_data[0]))


class FeatureExemplarsBuffer(ParametricBuffer):
    """Replay buffer that selects exemplars by feature-norm importance."""

    def __init__(self, max_size, model, layer_name, n_neighbors=10, groupby=None, selection_strategy=None):
        super().__init__(max_size, groupby, selection_strategy)
        self.feature_extractor = FeatureExtractorBackbone(model, layer_name)
        self.n_neighbors = n_neighbors
        print(">>>>FeatureExemplarsBuffer initialized with n_neighbors =", n_neighbors)

    def update(self, strategy, **kwargs):
        data_list = [
            MultiModal(
                embedding=data[0], path="", label=data[1], label_description="",
                zeroshot_label=-1, zeroshot_description="",
                task=data[2], task_description="",
                id_code="", metadata={}
            )
            for data in strategy.experience.dataset
        ]
        multimodal_data = DocList[MultiModal](data_list)

        features = self._extract_features(strategy, multimodal_data)
        sorted_indices = self._select_top_exemplars(features)

        def flatten(lst):
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

        sorted_indices = flatten(sorted_indices)
        selected_indices = sorted_indices[:self.n_neighbors]

        x_selected = [strategy.experience.dataset[i][0] for i in selected_indices]
        y_selected = [strategy.experience.dataset[i][1] for i in selected_indices]
        t_selected = [0 for _ in selected_indices]

        x_selected_pt = torch.stack(x_selected).to(strategy.device)
        t_selected_pt = torch.tensor(t_selected).to(strategy.device)

        strategy.model.eval()
        with torch.no_grad():
            y_selected_pt = strategy.model(x_selected_pt, t_selected_pt)
            y_selected_pred = torch.argmax(y_selected_pt, dim=1)

        new_data = TensorDataset(x_selected_pt.cpu(), y_selected_pred.cpu(), t_selected_pt.cpu())
        new_data = AvalancheDataset(new_data)
        new_data.targets = y_selected_pt.tolist()

        new_groups = self._make_groups(strategy, new_data)
        self.seen_groups.update(new_groups.keys())
        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {group_id: ll for group_id, ll in zip(self.seen_groups, lens)}

        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            if group_id in self.buffer_groups:
                old_buffer_g = self.buffer_groups[group_id]
                old_buffer_g.update_from_dataset(strategy, new_data_g)
                old_buffer_g.resize(strategy, ll)
            else:
                new_buffer = _ParametricSingleBuffer(ll, self.selection_strategy)
                new_buffer.update_from_dataset(strategy, new_data_g)
                self.buffer_groups[group_id] = new_buffer

        for group_id in self.buffer_groups:
            self.buffer_groups[group_id].resize(strategy, group_to_len[group_id])

    def _extract_features(self, strategy, multimodal_data):
        self.feature_extractor.eval()
        features = [self.feature_extractor(doc.embedding.to(strategy.device)) for doc in multimodal_data]
        return torch.stack(features, dim=0)

    def _select_top_exemplars(self, features):
        return self.make_sorted_indices_from_features(features)

    def make_sorted_indices_from_features(self, features: torch.Tensor) -> list:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        norms = torch.norm(features, dim=1)
        sorted_indices = torch.argsort(norms, descending=True)
        return sorted_indices.tolist() if torch.is_tensor(sorted_indices) else sorted_indices


class TADILER(SupervisedPlugin):
    """Avalanche plugin that injects CLIP-guided exemplar replay before each experience."""

    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy, num_workers: int = 0, shuffle: bool = False, **kwargs):
        if len(self.storage_policy.buffer) == 0:
            return

        print(f"strategy.adapted_dataset: {len(strategy.adapted_dataset[0])}, Length: {len(strategy.adapted_dataset)}")
        print(f"self.storage_policy.buffer: {len(self.storage_policy.buffer[0])}, Length: {len(self.storage_policy.buffer)}")
        print(f"strategy.train_mb_size: {strategy.train_mb_size}")

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle
        )

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy)
