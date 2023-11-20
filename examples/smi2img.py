from rdkit import Chem
from rdkit.Chem import Draw

if __name__ == "__main__":
    with open("mols.smi") as f:
        smis = f.readlines()

    smis = [smi.rstrip() for smi in smis]

    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = 5
    opts.minFontSize = 50

    for (idx, smi) in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        Draw.MolToFile(
            mol=mol,
            filename=f"sample_{idx}.png",
            size=(1024, 1024),
            options=opts,
        )
