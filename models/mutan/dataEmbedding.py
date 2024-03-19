import torch
import torch.nn as nn
import torch.nn.functional as F

class EggsDataNet(nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.embedding = nn.Embedding(6, inputDim)

        self.fc1 = nn.Linear(inputDim, outputDim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(outputDim, outputDim)
        self.dropout = nn.Dropout(0.5)

        # attention 机制 
        self.attentionFc = nn.Linear(inputDim, 1)
        # self.softmax = nn.Softmax()

    
    def forward(self, index, femaleAge, maleAge):
        output = self.embedding(index)

        output[:, 0, :].mul_(femaleAge.unsqueeze(-1).expand_as(output[:, 0, :]))
        output[:, 1, :].mul_(maleAge.unsqueeze(-1).expand_as(output[:, 1, :]))

        score0 = self.attentionFc(output[:, 0, :].squeeze())
        score1 = self.attentionFc(output[:, 1, :].squeeze())
        score2 = self.attentionFc(output[:, 2, :].squeeze())
        score3 = self.attentionFc(output[:, 3, :].squeeze())
        
        totalScore = torch.cat([score0, score1, score2, score3], dim=-1)
        
        # totalScore = self.softmax(totalScore)
        totalScore = torch.tanh(totalScore)
        totalScore = totalScore.unsqueeze(-1).expand_as(output)
        output = output * totalScore
        output = output.mean(dim=-2)

        # output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu1(output)
        # output = self.dropout(output)
        output = self.fc2(output)

        return output

if __name__ == '__main__':

    model = EggsDataNet(64,4)
    input = torch.tensor([[0,1,2,3], [0,1,4,5], [0,1,3,4]])
    femaleAge = torch.tensor([0.2, 0.5, 0.6])
    maleAge = torch.tensor([0.4, 0.6, 0.8])
    model(input, femaleAge, maleAge)
    assert False
    score0 = torch.tensor([[0], [0]])
    print(score0.size())
    score1 = torch.tensor([[1], [1]])
    score2 = torch.tensor([[2], [2]])
    score3 = torch.tensor([[3], [3]])
    data = torch.cat([score0, score1, score2, score3], dim=-1)
    print(data)
