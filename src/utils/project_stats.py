import os
from collections import defaultdict

# 支持统计的文件类型
EXTENSIONS = [
    '.py', '.yaml', '.yml', '.bat', '.sh', '.js', '.ts', '.client','.server'
]

def count_files_and_loc_by_type(root_dir, extensions=EXTENSIONS):
    stats = defaultdict(lambda: {'files': 0, 'loc': 0, 'details': []})
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loc = sum(1 for _ in f)
                except Exception:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            loc = sum(1 for _ in f)
                    except Exception:
                        loc = 0
                rel_path = os.path.relpath(file_path, root_dir)
                stats[ext]['files'] += 1
                stats[ext]['loc'] += loc
                stats[ext]['details'].append((rel_path, loc))
    return stats

def print_stats():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f'统计目录: {root}')
    stats = count_files_and_loc_by_type(root)
    total_files = sum(v['files'] for v in stats.values())
    total_loc = sum(v['loc'] for v in stats.values())
    print(f'支持类型文件总数: {total_files}')
    print(f'支持类型文件总行数(LOC): {total_loc}')
    print('\n按类型统计:')
    for ext, v in sorted(stats.items(), key=lambda x: -x[1]['loc']):
        print(f'  {ext}: 文件数={v["files"]}, 总行数={v["loc"]}')
        for path, loc in sorted(v['details'], key=lambda x: -x[1])[:5]:
            print(f'    {path}: {loc} 行')
        if len(v['details']) > 5:
            print(f'    ... 共{len(v["details"])}个文件')

if __name__ == '__main__':
    print_stats() 