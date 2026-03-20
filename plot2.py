def run_linked_data_experiment():
    ibge_map = get_ibge_reference()
    df_capes = get_capes_data()

    if df_capes is None:
        print("Failed to acquire CAPES data.")
        return

    # --- NEW: Dynamic Column Detection ---
    # Look for a column that likely contains the city name
    possible_cols = [col for col in df_capes.columns if 'MUNICIPIO' in col.upper()]
    
    if not possible_cols:
        print(f"Error: Could not find a municipality column. Available columns: {list(df_capes.columns)}")
        return
    
    target_col = possible_cols[0]
    print(f"-> Detected city column: '{target_col}'")

    # --- Proceed with Linkage ---
    df_capes['norm_city'] = df_capes[target_col].apply(normalize_text)
    df_capes['ibge_id'] = df_capes['norm_city'].map(ibge_map)
    
    # Calculate Results
    total = len(df_capes)
    linked = df_capes['ibge_id'].notna().sum()
    orphans = total - linked
    success_rate = (linked / total) * 100

    print(f"\nLinkage Success: {success_rate:.2f}%")
    
    # --- Visualization Code ---
    plt.figure(figsize=(10, 5))
    sns.barplot(x=['Linked', 'Orphan'], y=[linked, orphans], palette=['#2ecc71', '#e74c3c'])
    plt.title(f'Quality Analysis (Source Column: {target_col})')
    plt.savefig("results_plot.png")
    plt.show()

if __name__ == "__main__":
    run_linked_data_experiment()