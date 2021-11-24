import os

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import json
from PIL import Image


class ClassSpecificNumbering(torch.utils.data.Dataset):
    def __init__(self, set_size, set_dim, n_obj_per_sample, one_hot=True, rand_perm=False, n_samples=2**16):
        super().__init__()
        self.set_size = set_size
        self.set_dim = set_dim
        self.one_hot = one_hot
        self.rand_perm = rand_perm

        indices = torch.randint(0, n_obj_per_sample, (n_samples, set_size))
        values = torch.multinomial(torch.ones(n_samples, set_dim), n_obj_per_sample, replacement=False)
        inputs = torch.gather(values, dim=-1, index=indices)
        # u = inputs.unique()[None, None]
        u = torch.arange(0, set_dim)[None, None]
        c = (inputs.unsqueeze(-1) == u).int()  # indicator of size (n_samples, set_size, n_unique)
        targets = (c.cumsum(dim=1) * c).sum(dim=2) - 1  # 0 based

        self.inputs, self.targets = inputs, targets

    def __getitem__(self, index):
        i, t = self.inputs[index], self.targets[index]
        if self.one_hot:
            i = F.one_hot(i, num_classes=self.set_dim)
            t = F.one_hot(t, num_classes=len(t))
        if i.ndim < 2:
            i.unsqueeze_(1)
        if t.ndim < 2:
            t.unsqueeze_(1)
        if self.rand_perm:
            perm = torch.randperm(i.size(0))
            i = i[perm]
            t = t[perm]
        return i.float(), t.float()

    def __len__(self):
        return len(self.inputs)


class RandomMultisets(torch.utils.data.Dataset):
    def __init__(self, size=2**10, cardinality=10, dim=2):
        self.cardinality = cardinality
        self.size = size
        self.data = torch.randn(size, cardinality, dim)

    def __getitem__(self, item):
        coords = self.data[item]
        return coords, coords

    def __len__(self):
        return self.size


CLASSES = {
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
}


class CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path, split, image_input=False, image_size=128):
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.base_path = base_path
        self.split = split
        self.max_objects = 10
        self.image_input = image_input
        self.image_size = image_size
        if self.image_input:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.scenes = self.prepare_scenes(scenes)

    def object_to_fv(self, obj):
        coords = [p / 3 for p in obj["3d_coords"]]
        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        return coords + material + color + shape + size

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        for scene in scenes_json:
            img_idx = scene["image_index"]
            objects = [self.object_to_fv(obj) for obj in scene["objects"]]
            objects = torch.FloatTensor(objects)
            num_objects = objects.size(0)
            # pad with 0s
            if num_objects < self.max_objects:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(self.max_objects - num_objects, objects.size(1)),
                    ],
                    dim=0,
                )
            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            img_ids.append(img_idx)
            scenes.append((objects, mask))
        return img_ids, scenes

    @property
    def images_folder(self):
        return os.path.join(self.base_path, "images", self.split)

    @property
    def scenes_path(self):
        if self.split == "test":
            raise ValueError("Scenes are not available for test")
        return os.path.join(
            self.base_path, "scenes", "CLEVR_{}_scenes.json".format(self.split)
        )

    def load_image(self, image_id):
        filename = f'CLEVR_{self.split}_{image_id:06d}.png'
        path = os.path.join(self.images_folder, filename)
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image

    def __getitem__(self, item):
        image_id = self.img_ids[item]
        objects, mask = self.scenes[item]
        objects = torch.cat([objects, mask.unsqueeze(1)], dim=1)
        if not self.image_input:
            return objects, objects
        image = self.load_image(image_id)
        return image, objects

    def __len__(self):
        return len(self.scenes)
