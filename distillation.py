import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

def train_offline_kd(epoch, teacher_model, student_model, data_raw, data_grid, lr=0.00001, temperature=1.0, alpha=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_hard = nn.CrossEntropyLoss().to(device)
    criterion_soft = nn.KLDivLoss(reduction='batchmean').to(device)
    
    student_model.to(device)
    teacher_model.to(device)
    
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    student_model.train()  # only update student model parameters
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(zip(data_raw, data_grid)), total=len(data_raw), desc=f"Training Epoch {epoch}", leave=True)
    for i, (raw, grid) in progress_bar:
        # Unpacking raw and grid data
        X_raw, y_raw = raw
        X_grid, y_grid = grid
        
        X_raw, y_raw = X_raw.to(device), y_raw.to(device)
        X_grid, y_grid = X_grid.to(device), y_grid.to(device)
        
        assert torch.equal(y_raw, y_grid), "Both y must be equal"
        y = y_raw
        
        optimizer.zero_grad()

        # Forward pass teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(X_raw)

        # Forward pass student model
        student_outputs = student_model(X_grid)

        # Calculate the loss for hard label
        loss_hard = criterion_hard(student_outputs, y)

        # Calculate the loss for soft label
        loss_soft = criterion_soft(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1)
        )

        # Backpropagation
        loss = alpha * loss_soft + (1 - alpha) * loss_hard
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(student_outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # Update progress bar
        current_loss = running_loss / (i + 1)
        accuracy = 100 * correct / total
        progress_bar.set_postfix(Loss=f'{current_loss:.4f}', Accuracy=f'{accuracy:.2f}%')

    average_loss = running_loss / len(data_raw)
    accuracy = 100 * correct / total
    return average_loss, accuracy

def validate_offline_kd(epoch, teacher_model, student_model, data_raw, data_grid, temperature=1.0, alpha=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_hard = nn.CrossEntropyLoss().to(device)
    criterion_soft = nn.KLDivLoss(reduction='batchmean').to(device)
    
    student_model.to(device)
    teacher_model.to(device)
    
    student_model.eval()  # Set the student model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(zip(data_raw, data_grid)), total=len(data_raw), desc=f"Validation Epoch {epoch}", leave=True)
    with torch.no_grad():  # No gradients needed
        for i, (raw, grid) in progress_bar:
            # Unpacking raw and grid data
            X_raw, y_raw = raw
            X_grid, y_grid = grid
            
            X_raw, y_raw = X_raw.to(device), y_raw.to(device)
            X_grid, y_grid = X_grid.to(device), y_grid.to(device)
            
            assert torch.equal(y_raw, y_grid), "Both y must be equal"
            y = y_raw

            # Forward pass
            teacher_outputs = teacher_model(X_raw)
            student_outputs = student_model(X_grid)

            # Loss calculation
            loss_hard = criterion_hard(student_outputs, y)
            loss_soft = criterion_soft(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            )

            loss = alpha * loss_soft + (1 - alpha) * loss_hard

            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Update progress bar
            current_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total
            progress_bar.set_postfix(Loss=f'{current_loss:.4f}', Accuracy=f'{accuracy:.2f}%')

    average_loss = running_loss / len(data_raw)
    accuracy = 100 * correct / total
    return average_loss, accuracy

def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=True)
    with torch.no_grad():
        for i, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Update progress bar
            current_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total
            progress_bar.set_postfix(Loss=f'{current_loss:.4f}', Accuracy=f'{accuracy:.2f}%')

    average_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return average_loss, accuracy

def train_kd(rnd, epoch, teacher_model, student_model, data_raw, data_grid, lr=0.00001, temperature=1.0, alpha=0.5, mode='offline'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_hard = nn.CrossEntropyLoss().to(device)
    criterion_soft = nn.KLDivLoss(reduction='batchmean').to(device)
    
    student_model.to(device)
    teacher_model.to(device)
    
    optimizer_student = optim.Adam(student_model.parameters(), lr=lr)
    if mode == 'online':
        optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=lr)

    student_model.train()
    if mode == 'online':
        teacher_model.train()
    elif mode == 'offline':
        teacher_model.eval()

    running_loss_student = 0.0
    running_loss_teacher = 0.0
    correct_student = 0
    correct_teacher = 0
    total = 0

    progress_bar = tqdm(enumerate(zip(data_raw, data_grid)), total=len(data_raw), desc=f"Training Round {rnd} | Epoch {epoch+1}", leave=True)
    for i, (raw, grid) in progress_bar:
        # Unpacking raw and grid data
        X_raw, y_raw = raw
        X_grid, y_grid = grid
        
        X_raw, y_raw = X_raw.to(device), y_raw.to(device)
        X_grid, y_grid = X_grid.to(device), y_grid.to(device)
        
        assert torch.equal(y_raw, y_grid), "Both y must be equal"
        y = y_raw
        
        optimizer_student.zero_grad()
        if mode == 'online':
            optimizer_teacher.zero_grad()

        # Forward pass for both models
        teacher_outputs = teacher_model(X_raw)
        student_outputs = student_model(X_grid)

        # Calculate the hard loss
        loss_teacher_hard = criterion_hard(teacher_outputs, y)
        loss_student_hard = criterion_hard(student_outputs, y)

        # Calculate the soft loss
        loss_soft = criterion_soft(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1)
        )

        # Compute total losses
        loss_teacher = alpha * loss_soft.detach() + (1 - alpha) * loss_teacher_hard
        if mode == 'online':
            loss_teacher.backward(retain_graph=True)
        
        loss_student = alpha * loss_soft + (1 - alpha) * loss_student_hard
        if mode == 'online':
            loss_student.backward()
            
        # Backpropagation
        optimizer_student.step()
        if mode == 'online':
            optimizer_teacher.step()

        running_loss_student += loss_student.item()
        running_loss_teacher += loss_teacher.item()
        _, predicted_student = torch.max(student_outputs.data, 1)
        _, predicted_teacher = torch.max(teacher_outputs.data, 1)
        total += y.size(0)
        correct_student += (predicted_student == y).sum().item()
        correct_teacher += (predicted_teacher == y).sum().item()

        # Update progress bar
        current_loss_student = running_loss_student / (i + 1)
        current_loss_teacher = running_loss_teacher / (i + 1)
        accuracy_student = 100 * correct_student / total
        accuracy_teacher = 100 * correct_teacher / total
        progress_bar.set_postfix(Student_Loss=f'{current_loss_student:.4f}', Student_Accuracy=f'{accuracy_student:.2f}%', Teacher_Loss=f'{current_loss_teacher:.4f}', Teacher_Accuracy=f'{accuracy_teacher:.2f}%')

    average_loss_student = running_loss_student / len(data_raw)
    average_loss_teacher = running_loss_teacher / len(data_raw)
    accuracy_student = 100 * correct_student / total
    accuracy_teacher = 100 * correct_teacher / total
    return average_loss_student, average_loss_teacher, accuracy_student, accuracy_teacher

def validate_kd(rnd, epoch, teacher_model, student_model, data_raw, data_grid, temperature=1.0, alpha=0.5, mode='offline'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_hard = nn.CrossEntropyLoss().to(device)
    criterion_soft = nn.KLDivLoss(reduction='batchmean').to(device)
    
    student_model.to(device)
    teacher_model.to(device)
    
    student_model.eval()
    teacher_model.eval()
    
    running_loss_student = 0.0
    running_loss_teacher = 0.0
    correct_student = 0
    correct_teacher = 0
    total = 0

    progress_bar = tqdm(enumerate(zip(data_raw, data_grid)), total=len(data_raw), desc=f"Validation Round {rnd} | Epoch {epoch+1}", leave=True)
    with torch.no_grad():
        for i, (raw, grid) in progress_bar:
            # Unpacking raw and grid data
            X_raw, y_raw = raw
            X_grid, y_grid = grid
            
            X_raw, y_raw = X_raw.to(device), y_raw.to(device)
            X_grid, y_grid = X_grid.to(device), y_grid.to(device)
            
            assert torch.equal(y_raw, y_grid), "Both y must be equal"
            y = y_raw

            # Forward pass for both models
            teacher_outputs = teacher_model(X_raw)
            student_outputs = student_model(X_grid)

            # Loss calculation
            loss_teacher_hard = criterion_hard(teacher_outputs, y)
            loss_student_hard = criterion_hard(student_outputs, y)

            loss_soft = criterion_soft(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            )

            # Calculate total losses
            loss_student = alpha * loss_soft + (1 - alpha) * loss_student_hard
            running_loss_student += loss_student.item()

            # For teacher, we calculate but do not update
            loss_teacher = alpha * loss_soft.detach() + (1 - alpha) * loss_teacher_hard
            running_loss_teacher += loss_teacher.item()

            _, predicted_student = torch.max(student_outputs.data, 1)
            _, predicted_teacher = torch.max(teacher_outputs.data, 1)
            total += y.size(0)
            correct_student += (predicted_student == y).sum().item()
            correct_teacher += (predicted_teacher == y).sum().item()

            # Update progress bar with the latest losses and accuracies
            accuracy_student = 100 * correct_student / total
            accuracy_teacher = 100 * correct_teacher / total
            current_loss_student = running_loss_student / (i + 1)
            current_loss_teacher = running_loss_teacher / (i + 1)
            
            progress_bar.set_postfix(Student_Loss=f'{current_loss_student:.4f}', Student_Accuracy=f'{accuracy_student:.2f}%', Teacher_Loss=f'{current_loss_teacher:.4f}', Teacher_Accuracy=f'{accuracy_teacher:.2f}%')

    average_loss_student = running_loss_student / len(data_raw)
    average_loss_teacher = running_loss_teacher / len(data_raw)
    accuracy_student = 100 * correct_student / total
    accuracy_teacher = 100 * correct_teacher / total
    return average_loss_student, average_loss_teacher, accuracy_student, accuracy_teacher