import os

folders = [
    'backend',
    'backend/api',
    'backend/models', 
    'backend/services',
    'backend/utils',
    'training'
]

for folder in folders:
    init_path = os.path.join(folder, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            f.write('')
        print(f'Created: {init_path}')
    else:
        print(f'Already exists: {init_path}')

print('\nDone.')