import os
import zipfile
import shutil

def debug_dataset_structure():
    """
    Debug function to find and explore the dataset structure
    """
    print("=== DATASET DEBUGGING ===")
    
    # Check current directory
    print("Current directory contents:")
    current_files = os.listdir('.')
    for file in current_files:
        if 'dataset' in file.lower() or 'urdu' in file.lower() or '.zip' in file.lower():
            print(f"  📁 {file}")
    
    # Common paths to check
    paths_to_check = [
        'dataset',
        'dataset/dataset', 
        'urdu_ghazals_rekhta',
        'urdu_ghazals_rekhta/dataset',
        '/content/urdu_ghazals_rekhta',
        '/content/urdu_ghazals_rekhta/dataset',
        '/content/dataset',
        '/content/dataset/dataset'
    ]
    
    print("\nChecking common dataset paths:")
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"  ✅ Found: {path}")
            if os.path.isdir(path):
                contents = os.listdir(path)[:10]  # Show first 10 items
                print(f"     Contents: {contents}")
        else:
            print(f"  ❌ Not found: {path}")
    
    # Look for zip files
    print("\nLooking for zip files:")
    zip_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.zip') and ('urdu' in file.lower() or 'dataset' in file.lower()):
                zip_path = os.path.join(root, file)
                zip_files.append(zip_path)
                print(f"  📦 Found zip: {zip_path}")
    
    return zip_files

def extract_and_setup_dataset():
    """
    Extract dataset from zip file and set up proper structure
    """
    print("\n=== EXTRACTING DATASET ===")
    
    # Find zip files
    zip_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                zip_files.append(zip_path)
    
    if not zip_files:
        print("❌ No zip files found!")
        return None
    
    # Try to extract the most likely dataset zip
    for zip_path in zip_files:
        print(f"📦 Trying to extract: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List contents
                file_list = zip_ref.namelist()[:20]  # Show first 20 files
                print(f"   Zip contents (first 20): {file_list}")
                
                # Extract to current directory
                zip_ref.extractall('.')
                print(f"   ✅ Extracted successfully!")
                
                # Check what was extracted
                print("   Checking extracted contents...")
                for item in os.listdir('.'):
                    if os.path.isdir(item) and ('dataset' in item.lower() or 'urdu' in item.lower()):
                        print(f"   📁 Extracted directory: {item}")
                        subdirs = os.listdir(item)[:10]
                        print(f"      Subdirectories: {subdirs}")
                
                return True
                
        except Exception as e:
            print(f"   ❌ Failed to extract {zip_path}: {e}")
            continue
    
    return False

def find_correct_dataset_path():
    """
    Find the correct path to the dataset after extraction
    """
    print("\n=== FINDING DATASET PATH ===")
    
    possible_paths = []
    
    # Walk through all directories to find poet folders
    for root, dirs, files in os.walk('.'):
        # Look for directories that might contain poets
        if any(poet in dirs for poet in ['mirza-ghalib', 'ahmad-faraz', 'allama-iqbal']):
            print(f"✅ Found poet directories in: {root}")
            print(f"   Poets found: {[d for d in dirs if any(p in d for p in ['mirza', 'ahmad', 'allama', 'faiz'])]}")
            possible_paths.append(root)
    
    if possible_paths:
        # Return the most likely path
        best_path = possible_paths[0]
        print(f"🎯 Best dataset path: {best_path}")
        return best_path
    else:
        print("❌ No poet directories found!")
        return None

def verify_dataset_structure(dataset_path):
    """
    Verify the dataset has the expected structure
    """
    print(f"\n=== VERIFYING DATASET STRUCTURE ===")
    print(f"Checking path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Path doesn't exist!")
        return False
    
    poets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found {len(poets)} poet directories: {poets[:5]}...")  # Show first 5
    
    if len(poets) == 0:
        print("❌ No poet directories found!")
        return False
    
    # Check a sample poet directory
    sample_poet = poets[0]
    poet_path = os.path.join(dataset_path, sample_poet)
    print(f"\nChecking sample poet: {sample_poet}")
    
    subdirs = os.listdir(poet_path)
    print(f"Subdirectories: {subdirs}")
    
    # Check for 'ur' and 'en' directories
    if 'ur' in subdirs and 'en' in subdirs:
        ur_path = os.path.join(poet_path, 'ur')
        en_path = os.path.join(poet_path, 'en')
        
        ur_files = len(os.listdir(ur_path))
        en_files = len(os.listdir(en_path))
        
        print(f"✅ Found Urdu files: {ur_files}")
        print(f"✅ Found English files: {en_files}")
        
        # Show sample files
        if ur_files > 0:
            sample_ur_file = os.listdir(ur_path)[0]
            print(f"Sample Urdu file: {sample_ur_file}")
        
        return True
    else:
        print(f"❌ Missing 'ur' or 'en' directories. Found: {subdirs}")
        return False

def main():
    """
    Main function to debug and fix dataset issues
    """
    print("🔍 DATASET TROUBLESHOOTING TOOL")
    print("=" * 50)
    
    # Step 1: Debug current structure
    zip_files = debug_dataset_structure()
    
    # Step 2: Extract if needed
    if zip_files:
        print(f"\n💡 Found {len(zip_files)} zip file(s). Attempting extraction...")
        extract_and_setup_dataset()
    
    # Step 3: Find correct path
    correct_path = find_correct_dataset_path()
    
    # Step 4: Verify structure
    if correct_path:
        is_valid = verify_dataset_structure(correct_path)
        if is_valid:
            print(f"\n🎉 SUCCESS! Use this path: '{correct_path}'")
            print(f"\n📝 UPDATE YOUR CODE:")
            print(f"dataset_path = '{correct_path}'")
        else:
            print(f"\n❌ Dataset structure is invalid at {correct_path}")
    else:
        print(f"\n❌ Could not find valid dataset path")

if __name__ == "__main__":
    main()
