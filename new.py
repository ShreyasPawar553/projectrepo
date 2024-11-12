import pickle

# Load the fertilizer information
fertilizer_info = pickle.load(open('fertilizer.pkl', 'rb'))

# Check if 'classes_' attribute exists and print the classes
if hasattr(fertilizer_info, 'classes_'):
    all_classes = fertilizer_info.classes_
    print("Fertilizer Classes:")
    for class_name in all_classes:
        print(class_name)
else:
    print("No classes found in fertilizer_info.")
