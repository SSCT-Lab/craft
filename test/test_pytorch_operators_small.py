import pymongo
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import traceback
import json
from datetime import datetime

class PyTorchOperatorTesterSmall:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        Initialize the PyTorch operator tester for small scale testing
        
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
    
    def get_num_test_cases(self, document: Dict[str, Any]) -> int:
        """
        Get the number of test cases in a document
        
        Args:
            document: MongoDB document
            
        Returns:
            Number of test cases
        """
        # Find the maximum length among all parameter arrays
        max_len = 0
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def prepare_arguments(self, document: Dict[str, Any], case_index: int) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for a specific test case within a document
        
        Args:
            document: MongoDB document containing multiple test cases
            case_index: Index of the test case to prepare (0-based)
            
        Returns:
            Tuple of (positional_args, keyword_args)
        """
        args = []
        kwargs = {}
        api_name = document.get("api", "")
        
        # Handle *size parameter (for torch.empty, torch.zeros, torch.ones, etc.)
        if "*size" in document:
            size_data = document["*size"]
            if isinstance(size_data, list) and len(size_data) > case_index:
                size_value = size_data[case_index]
                # Convert size to proper format for torch.empty
                if isinstance(size_value, list):
                    # If it's a list like [2, 3], convert to tuple
                    if size_value:  # Non-empty list
                        args.append(tuple(size_value))
                    else:  # Empty list means scalar tensor
                        args.append(())
                elif isinstance(size_value, int):
                    # Single integer, convert to tuple
                    args.append((size_value,))
                else:
                    # Handle other cases
                    args.append(size_value)
        
        # Handle *tensors parameter (for torch.atleast_3d, torch.atleast_2d, etc.)
        if "*tensors" in document:
            tensors_data = document["*tensors"]
            if isinstance(tensors_data, list) and len(tensors_data) > case_index:
                tensor_list = tensors_data[case_index]
                if isinstance(tensor_list, list):
                    for tensor_data in tensor_list:
                        args.append(self.convert_to_tensor(tensor_data))
                else:
                    args.append(self.convert_to_tensor(tensor_list))
        
        # Handle condition parameter (for torch.where)
        if "condition" in document:
            condition_data = document["condition"]
            if isinstance(condition_data, list) and len(condition_data) > case_index:
                condition_tensor = self.convert_to_tensor(condition_data[case_index])
                args.append(condition_tensor)
        
        # Handle x parameter
        if "x" in document:
            x_data = document["x"]
            if isinstance(x_data, list) and len(x_data) > case_index:
                x_tensor = self.convert_to_tensor(x_data[case_index])
                args.append(x_tensor)
        
        # Handle y parameter
        if "y" in document:
            y_data = document["y"]
            if isinstance(y_data, list) and len(y_data) > case_index:
                y_tensor = self.convert_to_tensor(y_data[case_index])
                args.append(y_tensor)
        
        # Handle input parameter (for other operators)
        if "input" in document:
            input_data = document["input"]
            if isinstance(input_data, list) and len(input_data) > case_index:
                input_tensor = self.convert_to_tensor(input_data[case_index])
                args.append(input_tensor)
        
        # Handle other parameters
        for key, value in document.items():
            if key not in ["_id", "api", "condition", "x", "y", "input", "*size", "*tensors", "tensors", "out", "eigenvectors", "upper"]:
                if isinstance(value, list) and len(value) > 0:
                    # Get the value at case_index, or use the first value if index is out of range
                    idx = min(case_index, len(value) - 1)
                    param_value = value[idx]
                    if isinstance(param_value, dict):
                        param_value = self.convert_to_tensor(param_value)
                    kwargs[key] = param_value
        
        # Convert dtype and device parameters
        kwargs = self.convert_dtype_device_params(kwargs)
        
        # Validate dimension parameters for specific operators
        if api_name == "torch.diag_embed" and args:
            kwargs = self.validate_diag_embed_dims(args[0], kwargs)
        
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
                    elif len(parts) == 4:
                        # Handle torch.nn.functional.selu
                        module1 = getattr(torch, parts[1])  # torch.nn
                        module2 = getattr(module1, parts[2])  # torch.nn.functional
                        return getattr(module2, parts[3])  # torch.nn.functional.selu
            return None
        except AttributeError:
            return None
    
    def test_single_case(self, document: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """
        Test a single test case within a document
        
        Args:
            document: MongoDB document containing multiple test cases
            case_index: Index of the test case to test (0-based)
            
        Returns:
            Test result dictionary
        """
        api_name = document.get("api", "unknown")
        test_id = str(document.get("_id", "unknown"))
        
        result = {
            "test_id": test_id,
            "api": api_name,
            "case_index": case_index + 1,
            "status": "unknown",
            "error": None,
            "error_type": None,
            "traceback": None
        }
        
        # Skip deprecated functions
        if api_name in ["torch.symeig"]:
            result["status"] = "skipped"
            result["error"] = f"Operator {api_name} is deprecated and removed in current PyTorch version"
            result["error_type"] = "DeprecatedWarning"
            return result
        
        try:
            # Get the operator function
            op_func = self.get_operator_function(api_name)
            if op_func is None:
                result["status"] = "failed"
                result["error"] = f"Operator {api_name} not found"
                result["error_type"] = "AttributeError"
                return result
            
            # Prepare arguments for this specific case
            args, kwargs = self.prepare_arguments(document, case_index)
            
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
    
    def test_first_n_operators(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Test first N documents (operators) in the MongoDB collection (all test cases within each document)
        
        Args:
            n: Number of documents to test (default: 20)
            
        Returns:
            List of test results
        """
        print(f"Starting PyTorch operator testing (first {n} documents)...")
        print(f"Connected to MongoDB: {self.client.address}")
        
        # Get first N documents from the collection
        documents = list(self.collection.find().limit(n))
        
        print(f"Found {len(documents)} documents:")
        total_cases = 0
        for i, doc in enumerate(documents):
            api_name = doc.get("api", "unknown")
            num_cases = self.get_num_test_cases(doc)
            total_cases += num_cases
            print(f"  {i+1}. {api_name} ({num_cases} test cases)")
        
        print(f"\nTotal test cases to execute: {total_cases}")
        
        results = []
        passed_count = 0
        failed_count = 0
        
        current_case = 1
        for doc_idx, doc in enumerate(documents):
            api_name = doc.get("api", "unknown")
            num_cases = self.get_num_test_cases(doc)
            
            print(f"\n🔧 Testing operator {doc_idx+1}/{len(documents)}: {api_name} ({num_cases} cases)")
            
            operator_passed = 0
            operator_failed = 0
            
            # Test each case within this document
            for case_idx in range(num_cases):
                print(f"  Case {current_case}/{total_cases} (Op case: {case_idx+1}/{num_cases}): {api_name}")
                
                result = self.test_single_case(doc, case_idx)
                result["operator"] = api_name
                result["total_cases_for_operator"] = num_cases
                results.append(result)
                
                if result["status"] == "passed":
                    passed_count += 1
                    operator_passed += 1
                    print(f"    ✅ PASSED")
                elif result["status"] == "skipped":
                    print(f"    ⏭️ SKIPPED: {result['error']}")
                else:
                    failed_count += 1
                    operator_failed += 1
                    print(f"    ❌ FAILED: {result['error']}")
                
                current_case += 1
            
            print(f"  📊 {api_name} summary: {operator_passed} passed, {operator_failed} failed")
        
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        
        print(f"\n🎯 Overall Testing Results:")
        print(f"✅ Total Passed: {passed_count}")
        print(f"❌ Total Failed: {failed_count}")
        print(f"⏭️ Total Skipped: {skipped_count}")
        total_executed = passed_count + failed_count
        if total_executed > 0:
            print(f"📊 Success rate: {passed_count/total_executed*100:.1f}%")
        
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
            filename = f"pytorch_test_results_small_{timestamp}.json"
        
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
        print("=" * 80)
        
        # Group by operator first, then by error type
        operator_groups = {}
        for case in failed_cases:
            operator = case.get("operator", case.get("api", "Unknown"))
            if operator not in operator_groups:
                operator_groups[operator] = []
            operator_groups[operator].append(case)
        
        for operator, cases in operator_groups.items():
            print(f"\n🔧 {operator} ({len(cases)} failed cases):")
            
            # Group by error type within each operator
            error_groups = {}
            for case in cases:
                error_type = case.get("error_type", "Unknown")
                if error_type not in error_groups:
                    error_groups[error_type] = []
                error_groups[error_type].append(case)
            
            for error_type, error_cases in error_groups.items():
                print(f"  🔸 {error_type} ({len(error_cases)} cases):")
                for case in error_cases:
                    case_idx = case.get("case_index", "?")
                    total_cases = case.get("total_cases_for_operator", "?")
                    print(f"    - Case {case_idx}/{total_cases}: {case['error'][:100]}{'...' if len(case['error']) > 100 else ''}")
    
    def convert_dtype_device_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string dtype and device parameters to proper PyTorch objects
        
        Args:
            kwargs: Keyword arguments dictionary
            
        Returns:
            Updated kwargs with proper dtype and device objects
        """
        if "dtype" in kwargs:
            dtype_value = kwargs["dtype"]
            if isinstance(dtype_value, str):
                if dtype_value == "torchdtype":
                    kwargs["dtype"] = torch.float32  # Default dtype
                else:
                    dtype_map = {
                        "torch.float32": torch.float32,
                        "torch.float64": torch.float64,
                        "torch.int32": torch.int32,
                        "torch.int64": torch.int64,
                        "torch.bool": torch.bool,
                        "torch.uint8": torch.uint8
                    }
                    kwargs["dtype"] = dtype_map.get(dtype_value, torch.float32)
            elif isinstance(dtype_value, int):
                # Handle integer dtype codes (this seems to be a test data issue)
                # Map common integer codes to dtypes
                int_dtype_map = {
                    0: torch.float32,
                    1: torch.float64,
                    2: torch.int32,
                    3: torch.int64,
                    4: torch.bool,
                    5: torch.uint8,
                    8: torch.float32,  # Fallback for code 8
                }
                kwargs["dtype"] = int_dtype_map.get(dtype_value, torch.float32)
        
        if "device" in kwargs:
            device_str = kwargs["device"]
            if isinstance(device_str, str):
                if device_str == "cpu":
                    kwargs["device"] = torch.device("cpu")
                elif device_str.startswith("cuda"):
                    kwargs["device"] = torch.device(device_str)
        
        return kwargs
    
    def validate_diag_embed_dims(self, input_tensor: torch.Tensor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix dimension parameters for torch.diag_embed
        
        Args:
            input_tensor: Input tensor
            kwargs: Keyword arguments
            
        Returns:
            Updated kwargs with valid dimension parameters
        """
        input_ndim = len(input_tensor.shape)
        max_dim = input_ndim + 1
        
        if "dim1" in kwargs:
            dim1 = kwargs["dim1"]
            if isinstance(dim1, int):
                kwargs["dim1"] = max(-max_dim, min(dim1, max_dim - 1))
        
        if "dim2" in kwargs:
            dim2 = kwargs["dim2"]
            if isinstance(dim2, int):
                kwargs["dim2"] = max(-max_dim, min(dim2, max_dim - 1))
        
        return kwargs
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()

def main():
    """Main function to run the small scale tests"""
    # Initialize tester
    tester = PyTorchOperatorTesterSmall()
    
    try:
        # Run tests for first 20 operators
        results = tester.test_first_n_operators(20)
        
        # Save results
        tester.save_results(results)
        
        # Print error summary
        tester.print_error_summary(results)
        
        # Save failed cases separately if any
        failed_cases = tester.get_failed_cases(results)
        if failed_cases:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_filename = f"pytorch_failed_cases_small_{timestamp}.json"
            with open(failed_filename, 'w', encoding='utf-8') as f:
                json.dump(failed_cases, f, indent=2, ensure_ascii=False)
            print(f"Failed cases saved to: {failed_filename}")
        
    finally:
        # Close connection
        tester.close()

if __name__ == "__main__":
    main()
