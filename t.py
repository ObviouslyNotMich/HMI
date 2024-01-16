import os

def check_paths(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming each line starts with the file path, followed by space-separated values
            image_path = line.split()[0]

            # Check if the file exists
            if not os.path.isfile(image_path):
                print(f"File not found: {image_path}")

if __name__ == "__main__":
    check_paths('positives.txt')
