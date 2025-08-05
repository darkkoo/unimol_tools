from rdkit import Chem
from rdkit.Chem import AllChem

from unimol_tools.pretrain.preprocess import preprocess_dataset
from unimol_tools.pretrain.dataset import LMDBDataset


def test_preprocess_sdf(tmp_path):
    mol = Chem.MolFromSmiles('CCO')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    writer = Chem.SDWriter(str(tmp_path / 'mols.sdf'))
    writer.write(mol)
    writer.close()

    lmdb_path = tmp_path / 'mols.lmdb'
    preprocess_dataset(str(tmp_path / 'mols.sdf'), str(lmdb_path), data_type='sdf')

    dataset = LMDBDataset(str(lmdb_path))
    item = dataset[0]
    assert len(item['atoms']) == len(item['coordinates'])
    assert item['smi'] == Chem.MolToSmiles(mol)
