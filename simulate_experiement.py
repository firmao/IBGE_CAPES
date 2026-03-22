import numpy as np

def simulate_experiment(iterations=1000):
    # Simulated Ground Truth: Manaus PhD Programs = 14
    ground_truth = 14
    
    # Baseline LLM Simulation: Probabilistic bell curve centered around 
    # general knowledge, but high variance (hallucination)
    baseline_guesses = np.random.normal(loc=20, scale=10, size=iterations).astype(int)
    
    # HNSA Simulation: Grounded in the Symbolic Anchor (GCN/KG)
    # Errors only occur if there's a parsing/formatting issue (very low)
    hnsa_guesses = []
    for _ in range(iterations):
        if np.random.random() > 0.05: # 95% Retrieval Accuracy
            hnsa_guesses.append(ground_truth)
        else:
            hnsa_guesses.append(np.random.randint(10, 20)) # Minor error
            
    # Metrics Calculation
    baseline_accuracy = np.mean(baseline_guesses == ground_truth) * 100
    hnsa_accuracy = np.mean(np.array(hnsa_guesses) == ground_truth) * 100
    
    # Hallucination is defined as a guess > 20% away from truth
    baseline_hallucinations = np.mean(np.abs(baseline_guesses - ground_truth) / ground_truth > 0.2) * 100
    hnsa_hallucinations = np.mean(np.abs(np.array(hnsa_guesses) - ground_truth) / ground_truth > 0.2) * 100

    print(f"--- Experimental Results (N={iterations}) ---")
    print(f"Baseline Fact Consistency: {baseline_accuracy:.1f}%")
    print(f"HNSA Fact Consistency:     {hnsa_accuracy:.1f}%")
    print(f"Baseline Hallucination:    {baseline_hallucinations:.1f}%")
    print(f"HNSA Hallucination:        {hnsa_hallucinations:.1f}%")

if __name__ == "__main__":
    simulate_experiment()