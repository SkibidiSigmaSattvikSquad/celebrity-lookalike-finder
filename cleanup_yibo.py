#!/usr/bin/env python3
# remove yibo from database

import pickle
import os

cache_file = "encodings.pickle"

if os.path.exists(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            celeb_data = pickle.load(f)
        
        print(f"Loaded {len(celeb_data)} entries from database")
        
        original_count = len(celeb_data)
        cleaned_data = []
        
        for entry in celeb_data:
            name = entry.get('name', '').lower()
            img_path = entry.get('img_path', '').lower()
            
            if 'yibo' not in name and 'yibo' not in img_path:
                cleaned_data.append(entry)
            else:
                print(f"Removing entry: {entry.get('name', 'unknown')} ({entry.get('img_path', 'unknown')})")
        
        removed_count = original_count - len(cleaned_data)
        print(f"\nRemoved {removed_count} yibo entry/entries from database")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cleaned_data, f)
        
        print(f"✓ Database cleaned. Remaining entries: {len(cleaned_data)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("encodings.pickle not found - will be rebuilt on next run")

