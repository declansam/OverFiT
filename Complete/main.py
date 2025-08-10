import os
import torch
import torch.nn as nn

from GNNModels.models import GINConvNet
from MorganFingerprintMLP.models import MLP as MorganFingerprintMLP
from StructuralModels.models import StructureMLP
from Inference.gnn_predict import smiles_to_pyg_data

def smiles_to_graph(smiles):
    return smiles_to_pyg_data(smiles)

class StackedModel(nn.Module):

    def __init__(self):

        self.gnn_model = GINConvNet()
        self.morgan_mlp = MorganFingerprintMLP()
        self.structure_mlp = StructureMLP()

        self.meta_classifier = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, smiles):

        graph = smiles_to_graph(smiles)

        with torch.no_grad():

            morgan_pred = self.morgan_mlp(smiles)
            gnn_pred = self.gnn_model(graph)
            structure_pred = self.structure_mlp(smiles)

        individual_preds = torch.stack([gnn_pred, morgan_pred, structure_pred], dim=1)
        meta_pred = self.meta_classifier(individual_preds)
        return meta_pred

    @torch.no_grad()
    def predict(self, smiles):

        morgan_pred = self.morgan_mlp(smiles)

        graph = smiles_to_graph(smiles)
        gnn_pred = self.gnn_model(graph)
        structure_pred = self.structure_mlp(smiles)

        individual_preds = torch.stack([gnn_pred, morgan_pred, structure_pred], dim=1)
        meta_pred = self.meta_classifier(individual_preds)
        return meta_pred
        
    def load_models(self):

        self.gnn_model.load_state_dict(torch.load("GNNModels/gnn_model.pth"))
        self.morgan_mlp.load_state_dict(torch.load("MorganFingerprintMLP/morgan_mlp.pth"))
        self.structure_mlp.load_state_dict(torch.load("StructuralModels/structure_mlp.pth"))
        self.meta_classifier.load_state_dict(torch.load("MetaClassifier/meta_classifier.pth"))
    
    def save_models(self):

        torch.save(self.gnn_model.state_dict(), "GNNModels/gnn_model.pth")
        torch.save(self.morgan_mlp.state_dict(), "MorganFingerprintMLP/morgan_mlp.pth")
        torch.save(self.structure_mlp.state_dict(), "StructuralModels/structure_mlp.pth")
        torch.save(self.meta_classifier.state_dict(), "MetaClassifier/meta_classifier.pth")

def main():

    smiles = "C1=CC=C(C=C1)C(=O)O"

    model = StackedModel()
    model.load_models()
    model.eval()

    pred = model.predict(smiles)
    print(pred)

    model.save_models()

if __name__ == "__main__":
    main()