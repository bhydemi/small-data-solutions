import torch

def test_model(test_loader, trained_model, class_names):
    """Report overall and per-class accuracy on the held-out image split."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    correct = torch.zeros(len(class_names), dtype=torch.long)
    total = torch.zeros(len(class_names), dtype=torch.long)
    trained_model = trained_model.to(device).eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = trained_model(inputs).argmax(dim=1)
            for label, pred in zip(labels.cpu(), preds.cpu()):
                total[label] += 1
                correct[label] += int(label == pred)
    n = int(total.sum())
    accuracy = int(correct.sum()) / n if n else 0.0
    print(f"Test accuracy: {accuracy:.3%} ({int(correct.sum())}/{n})")
    for index, name in enumerate(class_names):
        score = int(correct[index]) / int(total[index]) if total[index] else 0.0
        print(f"{name}: {score:.3%} ({int(correct[index])}/{int(total[index])})")
    return accuracy
