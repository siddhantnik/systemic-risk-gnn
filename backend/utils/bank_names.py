import random

# Core Multinational Hubs
GLOBAL_HUBS = [
    "JPMorgan Chase", "HSBC Holdings", "Bank of America", "Citigroup",
    "Mitsubishi UFJ", "BNP Paribas", "Crédit Agricole", "Bank of China",
    "Barclays PLC", "Wells Fargo", "Société Générale", "UBS Group",
    "Deutsche Bank", "Goldman Sachs", "Morgan Stanley", "Santander",
    "Mizuho Financial", "Standard Chartered", "ING Group", "Credit Suisse"
]

# Mid-Tier / Regional
REGIONAL_PREFIXES = ["First", "National", "State", "Union", "Central", "Pacific", "Atlantic", "Northern", "Southern", "Western", "Eastern", "City", "Farmers", "Merchants", "Peoples"]
REGIONAL_SUFFIXES = ["Bank", "Trust", "Financial", "Bancorp", "Credit Union", "Capital"]

def generate_bank_name(node_idx: int, total_nodes: int, is_hub: bool = False, seed_offset: int = 42) -> str:
    """
    Deterministically generates a realistic string label for a specific bank index.
    
    Args:
        node_idx: The integer index of the node.
        total_nodes: Total nodes in the network (for scaling probabilities).
        is_hub: If True, draws from the GLOBAL_HUBS list.
    """
    random.seed(node_idx + seed_offset)
    
    if is_hub and node_idx < len(GLOBAL_HUBS):
        return GLOBAL_HUBS[node_idx]
    
    # Generate Regional/Local variations
    if random.random() > 0.8:
        # e.g., "Pacific Merchants Trust"
        name = f"{random.choice(REGIONAL_PREFIXES)} {random.choice(REGIONAL_PREFIXES)} {random.choice(REGIONAL_SUFFIXES)}"
    else:
        # e.g., "First National Bank"
        name = f"{random.choice(REGIONAL_PREFIXES)} {random.choice(REGIONAL_SUFFIXES)}"
        
    return f"{name} {node_idx}"
