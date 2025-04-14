import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TeacherModel(nn.Module):
    """Teacher model implementation for Mean-Teacher semi-supervised learning.
   
    Uses Exponential Moving Average (EMA) of student weights to generate stable 
    pseudo-labels for unlabeled data.
    
    Args:
        student_model: Model to create a teacher copy from
        ema_decay: Decay rate for exponential moving average (default: 0.99)
    """
    def __init__(self, student_model, ema_decay=0.99):
        super(TeacherModel, self).__init__()
        
        self.model = copy.deepcopy(student_model)
        self.ema_decay = ema_decay
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def update_weights(self, student_model):
        for teacher_param, student_param in zip(self.model.parameters(), student_model.parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data
    
    def generate_pseudo_labels(self, unlabeled_imgs, confidence_threshold=0.95):
        self.model.eval()
        with torch.no_grad():
            teacher_output = self.model(unlabeled_imgs)['out']
            probs = F.softmax(teacher_output, dim=1)
            confidence, pseudo_labels = torch.max(probs, dim=1)
 
            
            confidence_mask = confidence > confidence_threshold
            
            # print("probs", probs[0][0][0])
            # print("teacher_output", teacher_output.shape)
            # print("unlabeled_imgs", unlabeled_imgs.shape)
            # print("confidence", confidence)
            # print("confidence_mask", confidence_mask)
            
        return pseudo_labels, confidence_mask, confidence
