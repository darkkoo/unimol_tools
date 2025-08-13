import pickle
from functools import lru_cache

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from ..data.conformer import get_graph, get_graph_features

RDLogger.DisableLog('rdApp.*')


class LMDBDataset(Dataset):
    """
    Read LMDB and output idx, element types, atomic numbers, and 3D coordinates. Supports caching.
    """
    def __init__(self, lmdb_path):
        env = lmdb.open(
            lmdb_path, 
            subdir=False, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False, 
            max_readers=256,
        ) 
        self.txn = env.begin()
        self.length = self.txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.txn.get(str(idx).encode())
        if data is None:
            raise IndexError(f"Index {idx} not found in LMDB.")
        item = pickle.loads(data)
        atoms = item.get('atoms')
        coordinates = item.get('coordinates')

        result = {
            'idx': item.get('idx'),
            'atoms': atoms,
            'coordinates': coordinates,
            'smi': item.get('smi'),
        }
        return result

class UniMolDataset(Dataset):
    """
    Loads LMDBDataset for UniMol models.
    """
    def __init__(self, lmdb_dataset, dictionary, remove_hs=False, max_atoms=256, seed=1, sample_conformer=True, **params):
        self.dataset = lmdb_dataset
        self.length = len(self.dataset)
        self.dictionary = dictionary
        self.remove_hs = remove_hs
        self.max_atoms = max_atoms
        self.seed = seed
        self.sample_conformer = sample_conformer
        self.params = params
        self.mask_id = dictionary.add_symbol("[MASK]", is_special=True)
        self.set_epoch(0)  # Initialize epoch to 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
        np.random.seed(self.seed + epoch)
        self.sort_order = np.random.permutation(self.length)

    def ordered_indices(self):
        return self.sort_order


    def __getitem__(self, idx):
        return self.__getitem__cached__(self.epoch, idx)
    
    @lru_cache(maxsize=16)
    def __getitem__cached__(self, epoch, idx):
        item = self.dataset[idx]
        atoms = item['atoms']
        coordinates = item['coordinates']
        
        if atoms is None or coordinates is None:
            raise ValueError(f"Invalid data at index {idx}: atoms or coordinates are None.")
        
        if isinstance(coordinates, list):
            if self.sample_conformer:
                np.random.seed(self.seed + epoch + idx)
                sel = np.random.randint(len(coordinates))
                coordinates = coordinates[sel]
            else:
                coordinates = coordinates[0]
        
        net_input, target = coords2unimol(
            atoms=atoms, 
            coordinates=coordinates, 
            dictionary=self.dictionary, 
            mask_id=self.mask_id,
            noise_type=self.params.get('noise_type', 'trunc_normal'),
            noise=self.params.get('noise', 1.0),
            seed=self.params.get('seed', 1),
            epoch=epoch,
            mask_prob=self.params.get('mask_prob', 0.15),
            leave_unmasked_prob=self.params.get('leave_unmasked_prob', 0.1),
            random_token_prob=self.params.get('random_token_prob', 0.1),
            max_atoms=self.max_atoms,
            remove_hs=self.remove_hs,
        )
        return net_input, target

    
class UniMolV2Dataset(Dataset):
    """Dataset for UniMol2 pretraining without a dictionary."""

    def __init__(
        self,
        lmdb_dataset,
        remove_hs=False,
        max_atoms=256,
        seed=1,
        sample_conformer=True,
        mask_token_prob=0.15,
        drop_feat_prob=1.0,
        use_2d_pos_prob=0.5,
        noise_type="trunc_normal",
        noise=1.0,
    ):
        self.dataset = lmdb_dataset
        self.length = len(self.dataset)
        self.remove_hs = remove_hs
        self.max_atoms = max_atoms
        self.seed = seed
        self.sample_conformer = sample_conformer
        self.mask_token_prob = mask_token_prob
        self.drop_feat_prob = drop_feat_prob
        self.use_2d_pos_prob = use_2d_pos_prob
        self.noise_type = noise_type
        self.noise = noise
        self.mask_id = 127
        self.pad_idx = 0
        self.token_num = 128
        self.set_epoch(0)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
        np.random.seed(self.seed + epoch)
        self.sort_order = np.random.permutation(self.length)

    def ordered_indices(self):
        return self.sort_order

    def __getitem__(self, idx):
        return self.__getitem__cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem__cached__(self, epoch, idx):
        item = self.dataset[idx]
        atoms = item['atoms']
        coordinates = item['coordinates']
        if atoms is None or coordinates is None:
            raise ValueError(f"Invalid data at index {idx}: atoms or coordinates are None.")

        if isinstance(coordinates, list):
            if self.sample_conformer:
                np.random.seed(self.seed + epoch + idx)
                sel = np.random.randint(len(coordinates))
                coordinates = coordinates[sel]
            else:
                coordinates = coordinates[0]

        net_input, target, atom_idx = coords2unimol_v2(
            atoms=atoms,
            coordinates=coordinates,
            mask_id=self.mask_id,
            noise_type=self.noise_type,
            noise=self.noise,
            seed=self.seed,
            epoch=epoch,
            mask_token_prob=self.mask_token_prob,
            max_atoms=self.max_atoms,
            remove_hs=self.remove_hs,
            token_num=self.token_num,
        )
        graph_feat = build_graph_features(item['smi'], atom_idx, self.drop_feat_prob)
        net_input.update(graph_feat)
        return net_input, target

def coords2unimol(
        atoms, 
        coordinates, 
        dictionary, 
        mask_id,
        noise_type,
        noise=1.0,
        seed=1,
        epoch=0,
        mask_prob=0.15,
        leave_unmasked_prob=0.1,
        random_token_prob=0.1,
        max_atoms=256, 
        remove_hs=True, 
        **params
    ):
    np.random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)

    assert len(atoms) == len(coordinates), "coordinates shape does not align with atoms"
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        atoms, coordinates = atoms_no_h, coordinates_no_h

    # Crop atoms and coordinates if exceeding max_atoms
    if len(atoms) > max_atoms:
        idx = torch.randperm(len(atoms))[:max_atoms]
        atoms = [atoms[i] for i in idx.tolist()]
        coordinates = coordinates[idx]

    # Normalize coordinates
    coordinates = coordinates - coordinates.mean(dim=0)

    # Add noise and mask
    src_tokens, src_coord, tgt_tokens = apply_noise_and_mask(
        src_tokens=torch.tensor([dictionary.index(atom) for atom in atoms], dtype=torch.long),
        coordinates=coordinates,
        dictionary=dictionary,
        mask_id=mask_id,
        noise_type=noise_type,
        noise=noise,
        mask_prob=mask_prob,
        leave_unmasked_prob=leave_unmasked_prob,
        random_token_prob=random_token_prob
    )

    # Pad tokens
    src_tokens = torch.cat([torch.tensor([dictionary.bos()]), src_tokens, torch.tensor([dictionary.eos()])], dim=0)
    tgt_tokens = torch.cat([torch.tensor([dictionary.bos()]), tgt_tokens, torch.tensor([dictionary.eos()])], dim=0)

    # Pad coordinates
    pad = torch.zeros((1, 3), dtype=torch.float32)
    src_coord = torch.cat([pad, src_coord, pad], dim=0)
    tgt_coordinates = torch.cat([pad, coordinates, pad], dim=0)

    # Calculate distance matrix
    diff = src_coord.unsqueeze(0) - src_coord.unsqueeze(1)
    src_distance = torch.norm(diff, dim=-1)
    tgt_distance = torch.norm(tgt_coordinates.unsqueeze(0) - tgt_coordinates.unsqueeze(1), dim=-1)

    # Calculate edge type
    src_edge_type = src_tokens.view(-1, 1) * len(dictionary) + src_tokens.view(1, -1)

    return {
        'src_tokens': src_tokens,
        'src_coord': src_coord,
        'src_distance': src_distance,
        'src_edge_type': src_edge_type,
    },{
        'tgt_tokens': tgt_tokens,
        'tgt_coordinates': tgt_coordinates,
        'tgt_distance': tgt_distance,
    }
    
def coords2unimol_v2(
        atoms,
        coordinates,
        mask_id,
        noise_type,
        noise=1.0,
        seed=1,
        epoch=0,
        mask_token_prob=0.15,
        max_atoms=256,
        remove_hs=True,
        token_num=128,
    ):
    np.random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)

    assert len(atoms) == len(coordinates), "coordinates shape does not align with atoms"
    coordinates = torch.tensor(coordinates, dtype=torch.float32)

    idx = torch.arange(len(atoms))
    if remove_hs:
        keep = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms = [atoms[i] for i in keep]
        coordinates = coordinates[keep]
        idx = idx[keep]

    if len(atoms) > max_atoms:
        sel = torch.randperm(len(atoms))[:max_atoms]
        atoms = [atoms[i] for i in sel.tolist()]
        coordinates = coordinates[sel]
        idx = idx[sel]

    coordinates = coordinates - coordinates.mean(dim=0)

    src_tokens = torch.tensor([
        AllChem.GetPeriodicTable().GetAtomicNumber(a) for a in atoms
    ], dtype=torch.long)
    src_tokens, src_coord, tgt_tokens = apply_noise_and_mask_v2(
        src_tokens=src_tokens,
        coordinates=coordinates,
        mask_id=mask_id,
        noise_type=noise_type,
        noise=noise,
        mask_prob=mask_token_prob,
    )

    src_distance = torch.norm(src_coord.unsqueeze(0) - src_coord.unsqueeze(1), dim=-1)
    tgt_distance = torch.norm(coordinates.unsqueeze(0) - coordinates.unsqueeze(1), dim=-1)
    src_edge_type = src_tokens.view(-1, 1) * token_num + src_tokens.view(1, -1)

    return {
        'src_tokens': src_tokens,
        'src_coord': src_coord,
        'src_distance': src_distance,
        'src_edge_type': src_edge_type,
    }, {
        'tgt_tokens': tgt_tokens,
        'tgt_coordinates': coordinates,
        'tgt_distance': tgt_distance,
    }, idx


def build_graph_features(smiles, atom_idx, drop_feat_prob):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol, addCoords=True)
    mol = AllChem.RemoveAllHs(mol)
    node_attr, edge_index, edge_attr = get_graph(mol)
    drop_feat = np.random.rand() < drop_feat_prob
    mask = np.zeros(node_attr.shape[0], dtype=bool)
    mask[atom_idx.numpy()] = True
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat, mask)
    return {
        'atom_feat': torch.from_numpy(feat['atom_feat']).long(),
        'atom_mask': torch.from_numpy(feat['atom_mask']).long(),
        'edge_feat': torch.from_numpy(feat['edge_feat']).long(),
        'shortest_path': torch.from_numpy(feat['shortest_path']).long(),
        'degree': torch.from_numpy(feat['degree']).long(),
        'pair_type': torch.from_numpy(feat['pair_type']).long(),
        'attn_bias': torch.from_numpy(feat['attn_bias']).float(),
    }


def apply_noise_and_mask_v2(
        src_tokens,
        coordinates,
        mask_id,
        noise_type,
        noise=1.0,
        mask_prob=0.15,
    ):
    sz = len(src_tokens)
    num_mask = int(mask_prob * sz + np.random.rand())
    mask_idc = np.random.choice(sz, num_mask, replace=False)
    mask = np.zeros(sz, dtype=bool)
    mask[mask_idc] = True

    tgt_tokens = np.full(sz, 0)
    tgt_tokens[mask] = src_tokens.numpy()[mask]
    tgt_tokens = torch.from_numpy(tgt_tokens).long()

    mask_t = torch.from_numpy(mask)
    new_src_tokens = src_tokens.clone()
    new_src_tokens[mask_t] = mask_id

    num_mask = mask_t.sum().item()
    new_coordinates = coordinates.clone()
    if noise_type == "trunc_normal":
        noise_f = np.clip(np.random.randn(num_mask, 3) * noise, -2 * noise, 2 * noise)
    elif noise_type == "normal":
        noise_f = np.random.randn(num_mask, 3) * noise
    elif noise_type == "uniform":
        noise_f = np.random.uniform(-noise, noise, size=(num_mask, 3))
    else:
        noise_f = np.zeros((num_mask, 3), dtype=np.float32)
    new_coordinates[mask_t, :] += torch.tensor(noise_f, dtype=torch.float32)

    return new_src_tokens, new_coordinates, tgt_tokens


def apply_noise_and_mask(
        src_tokens, 
        coordinates,
        dictionary,
        mask_id,
        noise_type, 
        noise=1.0, 
        mask_prob=0.15, 
        leave_unmasked_prob=0.1, 
        random_token_prob=0.1
    ):
    """
    Apply noise and masking to the source tokens.
    """
    if random_token_prob > 0:
        weights = np.ones(len(dictionary)) 
        weights[dictionary.special_index()] = 0
        weights /= weights.sum()
    
    sz = len(src_tokens)
    assert sz > 0, "Source tokens must not be empty."

    num_mask = int(sz * mask_prob + np.random.rand())
    mask_idc = np.random.choice(sz, num_mask, replace=False)
    mask = np.full(sz, fill_value=False)
    mask[mask_idc] = True

    tgt_tokens = np.full(sz, dictionary.pad())
    tgt_tokens[mask] = src_tokens[mask]
    tgt_tokens = torch.from_numpy(tgt_tokens).long()

    # Determine unmasked and random tokens
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
        if random_token_prob == 0:
            unmasked = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0:
            unmasked = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            unmasked = rand_or_unmask & (np.random.rand(sz) < unmask_prob)
            rand_mask = rand_or_unmask & ~unmasked
    else:
        unmasked = None
        rand_mask = None
    
    if unmasked is not None:
        mask = mask ^ unmasked

    new_src_tokens = src_tokens.clone()
    new_src_tokens[mask] = mask_id

    num_mask = mask.sum().item()
    new_coordinates = coordinates.clone()

    # Add noise to masked coordinates
    if noise_type == "trunc_normal":
        noise_f = np.clip(np.random.randn(num_mask, 3) * noise, -noise*2, noise*2)
    elif noise_type == "normal":
        noise_f = np.random.randn(num_mask, 3) * noise
    elif noise_type == "uniform":
        noise_f = np.random.uniform(-noise, noise, size=(num_mask, 3))
    else:
        noise_f = np.zeros((num_mask, 3), dtype=np.float32)
    new_coordinates[mask, :] += torch.tensor(noise_f, dtype=torch.float32)

    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            new_src_tokens[rand_mask] = torch.tensor(
                np.random.choice(len(dictionary), num_rand, p=weights), dtype=torch.long
            )
    return new_src_tokens, new_coordinates, tgt_tokens