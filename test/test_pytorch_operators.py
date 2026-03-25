import pymongo
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import traceback
import json
from datetime import datetime

class PyTorchOperatorTester:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        Initialize the PyTorch operator tester
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        self.results = []
        
    def convert_to_tensor(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Convert MongoDB data to PyTorch tensor
        
        Args:
            data: Dictionary containing tensor information
            
        Returns:
            PyTorch tensor
        """
        if isinstance(data, dict):
            # Handle different data types
            dtype_map = {
                "torch.float64": torch.float64,
                "torch.float32": torch.float32,
                "torch.int64": torch.int64,
                "torch.int32": torch.int32,
                "torch.bool": torch.bool,
                "torch.uint8": torch.uint8
            }
            
            # Extract shape and dtype
            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, torch.float32)
            
            # Create tensor based on shape
            if shape:
                if dtype == torch.bool:
                    # For boolean tensors, create random boolean values
                    return torch.randint(0, 2, shape, dtype=dtype)
                elif dtype in [torch.int64, torch.int32]:
                    # For integer tensors, create random integers
                    return torch.randint(-10, 10, shape, dtype=dtype)
                else:
                    # For float tensors, create random floats
                    return torch.randn(shape, dtype=dtype)
            else:
                # Scalar tensor
                if dtype == torch.bool:
                    return torch.tensor(True, dtype=dtype)
                elif dtype in [torch.int64, torch.int32]:
                    return torch.tensor(1, dtype=dtype)
                else:
                    return torch.tensor(1.0, dtype=dtype)
        elif isinstance(data, (int, float)):
            return torch.tensor(data)
        elif isinstance(data, list):
            return torch.tensor(data)
        else:
            return torch.tensor(data)
    
    def prepare_arguments(self, test_case: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for the operator call
        
        Args:
            test_case: Test case data from MongoDB
            
        Returns:
            Tuple of (positional_args, keyword_args)
        """
        args = []
        kwargs = {}
        
        # Handle condition parameter (for torch.where)
        if "condition" in test_case:
            condition_data = test_case["condition"]
            if isinstance(condition_data, list) and len(condition_data) > 0:
                condition_tensor = self.convert_to_tensor(condition_data[0])
                args.append(condition_tensor)
        
        # Handle x parameter
        if "x" in test_case:
            x_data = test_case["x"]
            if isinstance(x_data, list) and len(x_data) > 0:
                x_tensor = self.convert_to_tensor(x_data[0])
                args.append(x_tensor)
        
        # Handle y parameter
        if "y" in test_case:
            y_data = test_case["y"]
            if isinstance(y_data, list) and len(y_data) > 0:
                y_tensor = self.convert_to_tensor(y_data[0])
                args.append(y_tensor)
        
        # Handle input parameter (for other operators)
        if "input" in test_case:
            input_data = test_case["input"]
            if isinstance(input_data, list) and len(input_data) > 0:
                input_tensor = self.convert_to_tensor(input_data[0])
                args.append(input_tensor)
        
        # Handle other parameters
        for key, value in test_case.items():
            if key not in ["_id", "api", "condition", "x", "y", "input", "tensors", "out", "eigenvectors", "upper"]:
                if isinstance(value, list) and len(value) > 0:
                    # Try to convert to appropriate type
                    param_value = value[0] if len(value) == 1 else value
                    if isinstance(param_value, dict):
                        param_value = self.convert_to_tensor(param_value)
                    kwargs[key] = param_value
        
        return args, kwargs
    
    def get_operator_function(self, api_name: str):
        """
        Get the PyTorch operator function from API name
        
        Args:
            api_name: API name like "torch.where"
            
        Returns:
            PyTorch function or None if not found
        """
        try:
            # Split the API name and get the function
            parts = api_name.split(".")
            if len(parts) >= 2:
                if parts[0] == "torch":
                    if len(parts) == 2:
                        return getattr(torch, parts[1])
                    elif len(parts) == 3:
                        # Handle cases like torch.nn.functional
                        module = getattr(torch, parts[1])
                        return getattr(module, parts[2])
            return None
        except AttributeError:
            return None
    
    def test_single_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single operator test case
        
        Args:
            test_case: Test case data from MongoDB
            
        Returns:
            Test result dictionary
        """
        api_name = test_case.get("api", "unknown")
        test_id = str(test_case.get("_id", "unknown"))
        
        result = {
            "test_id": test_id,
            "api": api_name,
            "status": "unknown",
            "error": None,
            "error_type": None,
            "traceback": None
        }
        
        try:
            # Get the operator function
            op_func = self.get_operator_function(api_name)
            if op_func is None:
                result["status"] = "failed"
                result["error"] = f"Operator {api_name} not found"
                result["error_type"] = "AttributeError"
                return result
            
            # Prepare arguments
            args, kwargs = self.prepare_arguments(test_case)
            
            # Execute the operator
            with torch.no_grad():  # Disable gradient computation for testing
                output = op_func(*args, **kwargs)
                result["status"] = "passed"
                result["output_shape"] = list(output.shape) if hasattr(output, 'shape') else None
                result["output_dtype"] = str(output.dtype) if hasattr(output, 'dtype') else None
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def test_all_operators(self) -> List[Dict[str, Any]]:
        """
        Test all operators in the MongoDB collection
        
        Returns:
            List of test results
        """
        print("Starting PyTorch operator testing...")
        print(f"Connected to MongoDB: {self.client.address}")
        
        # Get all documents from the collection
        documents = list(self.collection.find())
        print(f"Found {len(documents)} test cases to execute")
        
        results = []
        passed_count = 0
        failed_count = 0
        
        for i, doc in enumerate(documents):
            print(f"Testing case {i+1}/{len(documents)}: {doc.get('api', 'unknown')}")
            
            result = self.test_single_case(doc)
            results.append(result)
            
            if result["status"] == "passed":
                passed_count += 1
            else:
                failed_count += 1
                print(f"  ❌ FAILED: {result['error']}")
        
        print(f"\nTesting completed!")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📊 Success rate: {passed_count/(passed_count+failed_count)*100:.1f}%")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """
        Save test results to a JSON file
        
        Args:
            results: Test results
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pytorch_test_results_{timestamp}.json"
        
        # Prepare summary
        passed_count = sum(1 for r in results if r["status"] == "passed")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        
        output_data = {
            "summary": {
                "total_tests": len(results),
                "passed": passed_count,
                "failed": failed_count,
                "success_rate": passed_count / len(results) * 100 if results else 0,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filename}")
    
    def get_failed_cases(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get only the failed test cases
        
        Args:
            results: All test results
            
        Returns:
            List of failed test cases
        """
        return [r for r in results if r["status"] == "failed"]
    
    def print_error_summary(self, results: List[Dict[str, Any]]):
        """
        Print a summary of errors
        
        Args:
            results: Test results
        """
        failed_cases = self.get_failed_cases(results)
        
        if not failed_cases:
            print("🎉 All test cases passed!")
            return
        
        print(f"\n📋 Error Summary ({len(failed_cases)} failed cases):")
        print("=" * 60)
        
        # Group by error type
        error_groups = {}
        for case in failed_cases:
            error_type = case.get("error_type", "Unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(case)
        
        for error_type, cases in error_groups.items():
            print(f"\n🔸 {error_type} ({len(cases)} cases):")
            for case in cases[:5]:  # Show first 5 cases of each error type
                print(f"  - {case['api']}: {case['error']}")
            if len(cases) > 5:
                print(f"  ... and {len(cases) - 5} more cases")
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()

def main():
    """Main function to run the tests"""
    # Initialize tester
    tester = PyTorchOperatorTester()
    
    try:
        # Run all tests
        results = tester.test_all_operators()
        
        # Save results
        tester.save_results(results)
        
        # Print error summary
        tester.print_error_summary(results)
        
        # Save failed cases separately
        failed_cases = tester.get_failed_cases(results)
        if failed_cases:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_filename = f"pytorch_failed_cases_{timestamp}.json"
            with open(failed_filename, 'w', encoding='utf-8') as f:
                json.dump(failed_cases, f, indent=2, ensure_ascii=False)
            print(f"Failed cases saved to: {failed_filename}")
        
    finally:
        # Close connection
        tester.close()

if __name__ == "__main__":
    main()
