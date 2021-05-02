from data.dataset import MoleculeDataset

if __name__ == "__main__":
    for dataset_name in ["bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv"]:
        print(dataset)
        dataset = MoleculeDataset("../resource/dataset/" + dataset_name, dataset=dataset_name)
        smiles_list = pd.read_csv(
            '../resource/dataset/' + dataset_name + '/processed/smiles.csv', header=None
            )[0].tolist()

        for idx in range(len(dataset)):
            data = dataset[idx]
            smiles = smiles_list[idx]

            print(smiles)
            print(data.y.tolist())
            assert False
        