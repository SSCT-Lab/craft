#!/usr/bin/env python3
"""
Test script to verify the torch.empty fix
"""

import torch
import sys
import os

# Add the current directory to path to import the test module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_pytorch_operators_small import PyTorchOperatorTesterSmall

def test_torch_empty_cases():
    """Test various torch.empty cases that were failing"""
    
    # Create a mock document similar to the MongoDB structure
    mock_document = {
        "_id": "test_id",
        "api": "torch.empty",
        "*size": [
            [],           # Empty list -> scalar tensor
            1,            # Single int -> (1,)
            5,            # Single int -> (5,)
            [2, 3],       # List -> (2, 3)
            [1, 0],       # List with 0 -> (1, 0)
            [0, 2]        # List starting with 0 -> (0, 2)
        ],
        "dtype": [
            "torchdtype",  # Default dtype
            "torch.float32",
            "torch.int64",
            8,             # Integer dtype code
            "torch.bool",
            "torch.float64"
        ],
        "device": [
            "cpu",
            "cpu",
            "cpu",
            "cpu",
            "cpu",
            "cpu"
        ]
    }
    
    # Initialize tester
    tester = PyTorchOperatorTesterSmall()
    
    print("Testing torch.empty fixes...")
    print("=" * 50)
    
    # Test each case
    for case_idx in range(6):
        print(f"\nTest case {case_idx + 1}:")
        
        try:
            # Prepare arguments
            args, kwargs = tester.prepare_arguments(mock_document, case_idx)
            print(f"  Args: {args}")
            print(f"  Kwargs: {kwargs}")
            
            # Try to call torch.empty
            result = torch.empty(*args, **kwargs)
            print(f"  ✅ SUCCESS: Created tensor with shape {result.shape}, dtype {result.dtype}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_torch_empty_cases()
