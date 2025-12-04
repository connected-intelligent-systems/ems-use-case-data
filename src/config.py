from pathlib import Path

total_path_clearml = "uri:urn:554b7c6c19254be5/power.csv"
devices_dict_clearml = {
        "DECT210 Waschmaschine": "uri:urn:6a112240-117e-4ec3-a129-5bc90908aedb",    
        "DECT200 Waschmaschine": "uri:urn:4b29c04c920141e8",
        "Smart Switch 6 Spülmaschine": "uri:urn:e91f9319-71af-4ddb-ab7d-fb47b45d69ad",    
        "DECT200 Spülmaschine": "uri:urn:cc256ae649904024",
    }

saving_dir = Path("./intermediate_saves/")
save_intermediates = True
use_cached = True

process_freq = '10s'
output_freq = '15min'