"""
Extract all PyTorch operator names from MongoDB and save to a CSV file.

Database: freefuzz-torch
Collection: argVS
Output: pytorch_apis.csv (column: pytorch-api)
"""

import os
import csv
from typing import List, Set

import pymongo


def extract_pytorch_apis(
    mongo_uri: str = "mongodb://localhost:27017/",
    db_name: str = "freefuzz-torch",
    collection_name: str = "argVS"
) -> List[str]:
    """
    Extract all PyTorch operator names from MongoDB.

    Args:
        mongo_uri: MongoDB connection URI
        db_name: database name
        collection_name: collection name

    Returns:
        Deduplicated operator list (alphabetically sorted)
    """
    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Use distinct to get unique API values (more efficient than scanning all docs)
    api_names: List[str] = collection.distinct("api")
    
    # Filter empty values
    api_names = [api for api in api_names if api]
    
    # Sort alphabetically
    api_names.sort()
    
    # Close connection
    client.close()
    
    return api_names


def save_to_csv(api_names: List[str], output_path: str) -> None:
    """
    Save operator names to a CSV file.

    Args:
        api_names: operator name list
        output_path: output file path
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['pytorch-api'])
        # Write data
        for api_name in api_names:
            writer.writerow([api_name])


def main():
    """Main entry point."""
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'api_mappings.csv')
    
    print("=" * 50)
    print("📊 PyTorch operator name extractor")
    print("=" * 50)
    
    try:
        # Extract operator names
        print("\n🔗 Connecting to MongoDB...")
        api_names = extract_pytorch_apis()
        
        print(f"✅ Extracted {len(api_names)} unique operator names")
        
        # Save to CSV
        save_to_csv(api_names, output_path)
        print(f"✅ Saved to: {output_path}")
        
        # Preview first 10 operators
        print("\n📋 Operator preview (first 10):")
        print("-" * 30)
        for i, api_name in enumerate(api_names[:10], 1):
            print(f"  {i}. {api_name}")
        
        if len(api_names) > 10:
            print(f"  ... ({len(api_names)} total)")
        
        print("\n" + "=" * 50)
        print("🎉 Done!")
        print("=" * 50)
        
    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   Please ensure the MongoDB service is running")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
