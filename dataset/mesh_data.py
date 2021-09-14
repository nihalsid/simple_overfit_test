import trimesh
from torch.utils.data import Dataset
import torch
import numpy as np


class MeshData(Dataset):

    def __init__(self):
        super().__init__()
        self.mesh = trimesh.load('dataset/data/model_normalized.obj', force='mesh', process=False)
        self.vertex_max = self.mesh.vertices.max()
        self.vertex_min = self.mesh.vertices.min()

    def __len__(self):
        return self.mesh.vertices.shape[0]

    def __getitem__(self, index):
        v = self.mesh.vertices[index, :]
        # colors between 0 and 1
        c = self.mesh.visual.vertex_colors[index, :3] / 255 - 0.5
        # vertices between -1 and 1
        v = -1 + 2 * ((v - self.vertex_min) / (self.vertex_max - self.vertex_min))
        return {
            'vertex': np.array(v).astype(np.float32),
            'color': np.array(c).astype(np.float32)
        }

    @staticmethod
    def move_batch_to_gpu(batch, device):
        batch['vertex'] = batch['vertex'].to(device)
        batch['color'] = batch['color'].to(device)

    def get_data_on_device(self, device):
        v = self.mesh.vertices[:, :]
        # colors between 0 and 1
        c = self.mesh.visual.vertex_colors[:, :3] / 255 - 0.5
        # vertices between -1 and 1
        v = -1 + 2 * ((v - self.vertex_min) / (self.vertex_max - self.vertex_min))
        return torch.from_numpy(v).float().to(device), torch.from_numpy(c).float().to(device)

    def visualize(self, prediction, outdir, epoch):
        prediction = np.clip(prediction, -0.5, 0.5)
        mesh = trimesh.Trimesh(vertices=self.mesh.vertices, faces=self.mesh.faces, vertex_colors=prediction + 0.5, process=False)
        mesh.export(f"{outdir}/{epoch:05d}_pred.obj")
        mesh = trimesh.Trimesh(vertices=self.mesh.vertices, faces=self.mesh.faces, vertex_colors=self.mesh.visual.vertex_colors[:, :3] / 255, process=False)
        mesh.export(f"{outdir}/{epoch:05d}_gt.obj")
