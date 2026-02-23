import os, time, json, math, hashlib, re, pickle, subprocess, shutil, types, struct, sys, traceback, inspect
NP = False
try:
    from plic_vec import vec_norm, vec_dot
    print(' Loaded lightning-fast C vector ops! ')
except ImportError as e:
    print(f'C extension failed to load: {e} — using pure Python fallback')
try:
    import psutil
except ImportError:
    psutil = None
    from collections import deque
SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
EVOLUTION_DIR = os.path.join(SANDBOX_DIR, 'evolution')
os.makedirs(EVOLUTION_DIR, exist_ok=True)
SEEN_LOG = os.path.join(SANDBOX_DIR, 'seen_files.json')
seen_files = json.load(open(SEEN_LOG)) if os.path.exists(SEEN_LOG) else {}

class SelfReader:

    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.code_snapshot = ''
        self.log_snapshot = ''

    def read_code(self):
        try:
            with open(__file__, 'r', encoding='utf-8', errors='replace') as f:
                self.code_snapshot = f.read()
        except:
            self.code_snapshot = ''

    def read_logs(self):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                self.log_snapshot = f.read()
        except:
            self.log_snapshot = ''
self_reader = SelfReader(SANDBOX_DIR)

def resize_vector(v, target_len, fill=0.0):
    return v[:min(len(v), target_len)] + [fill] * (target_len - len(v))

def get_signature(func):
    return inspect.signature(func)

def auto_func_phase_gate(C, T, M, S, B):
    try:
        coh_ct = vec_cos_sim(C, T)
        linear_mix = vec_add(C, T, M)
        diff = vec_sub(S, T)
        mag = vec_norm(C) + 1e-09
        stability = 1.0 / (1.0 + mag)
        mix_factor = max(0.0, min(1.0, coh_ct - 0.2)) / (0.8 + 1e-09)
        out = vec_add(vec_mul_scalar(linear_mix, mix_factor), vec_mul_scalar(diff, 1.0 - mix_factor))
        out = vec_mul_scalar(out, stability)
        return vec_clip(out, -100, 100)
    except Exception:
        return C

def should_read(path):
    size = os.path.getsize(path)
    if seen_files.get(path) == size:
        return False
    seen_files[path] = size
    print(f'[DEBUG] Seen log loaded: {len(seen_files)} files')
    return True
os.makedirs(SANDBOX_DIR, exist_ok=True)
open(os.path.join(SANDBOX_DIR, 'success_log.txt'), 'a').write('=== Evolution success log ===\nSeeded at startup\n')
EVOLUTION_LOG = os.path.join(EVOLUTION_DIR, 'evolution_attempts.jsonl')
if not os.path.exists(EVOLUTION_LOG):
    with open(EVOLUTION_LOG, 'w', encoding='utf-8', errors='replace') as f:
        f.write('')
MUT_MEM_FILE = os.path.join(SANDBOX_DIR, 'mutation_memory.json')
try:
    mutation_memory = json.load(open(MUT_MEM_FILE))
except:
    mutation_memory = {}

def record_mutation(code_text, delta):
    h = hashlib.md5(code_text.encode('utf-8', errors='replace')).hexdigest()
    mutation_memory[h] = delta

def save_mutation_memory():
    json.dump(mutation_memory, open(MUT_MEM_FILE, 'w'))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
MAX_CYCLES = 1
current_cycle = 0
GLOBAL_DIM = 64
DIM = GLOBAL_DIM
global_steps = 100000
SELF_ORG_RATE = 0.36
NOISE_SCALE = 0.0036
C_FILE = os.path.join(SANDBOX_DIR, 'carrier.npy' if NP else 'carrier.json')
CONTINUITY_FILE = os.path.join(SANDBOX_DIR, 'continuity.npy' if NP else 'continuity.json')
SEMANTIC_FILE = os.path.join(SANDBOX_DIR, 'semantic.npy' if NP else 'semantic.json')
LOG_FILE = os.path.join(SANDBOX_DIR, 'plic_v4_1_log.jsonl')
RUNS_SUMMARY = os.path.join(SANDBOX_DIR, 'runs_summary.jsonl')
SEM_PERSIST_THRESH = 0.0005
EXPLORE_BASE = 0.02
EXPLORE_MAX = 0.08
EXPLORE_DECAY = 0.96
SEM_LEARN_MIN, SEM_LEARN_MAX = (0.002, 0.18)
SEM_THRESH_MIN, SEM_THRESH_MAX = (1e-05, 0.01)
ENTROPY_CEILING_MIN, ENTROPY_CEILING_MAX = (10.0, 300.0)
ENTROPY_CEILING = 81.0
DEBUG_LOG_FILE = os.path.join(SANDBOX_DIR, 'mutation_debug.log')
FAIL_LOG_FILE = os.path.join(SANDBOX_DIR, 'fail_log.txt')
SUCCESS_LOG_FILE = os.path.join(SANDBOX_DIR, 'success_log.txt')
snippet = ''
last_fitness_delta = 0.0

def env_sonar_ping(drive_root, max_files=100000):
    """Deterministic scan of the drive for structure (dirs, files, sizes, empty space). Returns a vector signature."""
    env_hashes = []
    count = 0
    for root, dirs, files in os.walk(drive_root):
        if count > max_files:
            break
        dir_hash = hashlib.md5(''.join(dirs).encode()).digest()
        env_hashes.append(int.from_bytes(dir_hash, 'little') % DIM)
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path) if os.path.isfile(path) else 0
            size_hash = hashlib.md5(str(size).encode()).digest()
            env_hashes.append(int.from_bytes(size_hash, 'little') % DIM)
            count += 1
            if count > max_files:
                break
    env_vec = [0.0] * DIM
    for h in env_hashes:
        env_vec[h] += 1.0
    return normalize(env_vec)

def plic_controlled_gate(state, size, target_wires, control_mask, control_wires, inverse, params):
    gamma = params[0] if params else 1.0

    def phi(x):
        n = math.sqrt(sum((v * v for v in x)))
        return None if n < 1e-12 else [v / n for v in x]

    def coh(x, y):
        px, py = (phi(x), phi(y))
        if px is None or py is None:
            return 0.0
        return sum((a * b for a, b in zip(px, py)))

    def P(x):
        n = math.sqrt(sum((v * v for v in x)))
        return [v * (1.0 + n) for v in x]
    for i in range(len(state)):
        if any((i >> w & 1 != int(m) for w, m in zip(control_wires, control_mask))):
            continue
        for t in target_wires:
            if i >> t & 1:
                x = state[i]
                y = state[i]
                if coh(x, y) > 0.999:
                    state[i] = P(x)
                else:
                    state[i] = [a + gamma * (a - b) for a, b in zip(x, y)]
    return state

def rotation_nd(N, theta, i, j):
    """
    Creates an NxN complex unitary rotation 
    in the plane spanned by basis vectors i and j.
    """
    U = np.eye(N, dtype=complex)
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -1j * s
    U[j, i] = -1j * s
    return U

def phase_shift(N, k, phi):
    """
    Applies a phase shift to basis state k.
    """
    U = np.eye(N, dtype=complex)
    U[k, k] = np.exp(1j * phi)
    return U

def build_circuit(N, rotation_params, phase_params=None):
    """
    N: dimension of space

    rotation_params = [
        (theta1, i1, j1),
        (theta2, i2, j2),
        ...
    ]

    phase_params = [
        (k1, phi1),
        (k2, phi2),
        ...
    ]
    """
    U = np.eye(N, dtype=complex)
    for theta, i, j in rotation_params:
        U = rotation_nd(N, theta, i, j) @ U
    if phase_params:
        for k, phi in phase_params:
            U = phase_shift(N, k, phi) @ U
    return U

def quantum_fitness(code_text_or_params):
    params = code_text_or_params if isinstance(code_text_or_params, list) else [0.0]
    U_c = build_circuit(params)
    U_target = [[0, 1], [1, 0]]
    diff = sum((sum((abs(U_c[i][j] - U_target[i][j]) ** 2 for j in range(2))) for i in range(2)))
    qfit = -math.sqrt(diff)
    return qfit

def micro_couple(micro_vec, macro_vec, coupling=0.01):
    coh = sum((mi * ma for mi, ma in zip(micro_vec, macro_vec))) / max(sum((mi ** 2 for mi in micro_vec)) ** 0.5 * sum((ma ** 2 for ma in macro_vec)) ** 0.5, 1e-21)
    return [mv + coupling * coh * mv for mv in macro_vec]

def get_hardware_stress(dim=DIM):
    if not psutil:
        return vec_zero(dim)
    cpu = psutil.cpu_percent() / 100.0
    mem = psutil.virtual_memory().percent / 100.0
    io = psutil.disk_io_counters().read_bytes % 10 ** 9 / 10 ** 9
    h_vec = vec_zero(dim)
    h_vec[0], h_vec[1], h_vec[2] = (cpu, mem, io)
    return h_vec

def log_evolution_attempt(action, details=''):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f'{timestamp} | {action} | {details.strip()}\n'
    try:
        os.makedirs(os.path.dirname(EVOLUTION_LOG), exist_ok=True)
        with open(EVOLUTION_LOG, 'a', encoding='utf-8', buffering=1) as f:
            f.write(entry)
            f.flush()
    except PermissionError:
        print(f'[LOG ERROR] Permission denied writing to {EVOLUTION_LOG}')
    except OSError as e:
        print(f'[LOG ERROR] OS error writing evolution log: {e}')
    except Exception as e:
        print(f'[LOG ERROR] Unexpected error in log_evolution_attempt: {e}')

def vec_to_str(vec):
    return ','.join((f'{x:.6f}' for x in vec))

def log_emergent(msg):
    print(f'Emergent log: {msg}')

def vec_zero(dim=DIM):
    return np.zeros(dim) if NP else [0.0] * dim
S = S if 'S' in globals() else [0]

def vec_rand(dim=DIM, scale=1.0):
    return [math.sin(i * math.pi / dim + S[0] * 3) * scale for i in range(dim)]

def vec_add(*vecs):
    """Add any number of vectors element-wise"""
    if not vecs:
        return []
    result = vecs[0][:]
    for v in vecs[1:]:
        result = [x + y for x, y in zip(result, v)]
    return result

def vec_sub(*vecs):
    """Subtract multiple vectors element-wise: v1 - v2 - v3 ..."""
    if not vecs:
        return []
    result = vecs[0][:]
    for v in vecs[1:]:
        if len(v) != len(result):
            raise ValueError('All vectors must be the same length')
        result = [x - y for x, y in zip(result, v)]
    return result

def vec_mul_scalar(vec, *scalars):
    """Multiply vec by product of scalars"""
    factor = 1.0
    for s in scalars:
        factor *= s
    return [x * factor for x in vec]

def vec_norm(a):
    return float(np.linalg.norm(a)) if NP else math.sqrt(sum((x * x for x in a)))

def vec_clip(a, low=-1000000000.0, high=1000000000.0):
    if NP:
        return np.clip(a, low, high)
    else:
        return [max(min(x, high), low) for x in a]

def normalize(v):
    norm = math.sqrt(sum((x * x for x in v))) + 1e-09
    return [x / norm for x in v]

def vec_copy(a):
    return a.copy() if NP else list(a)

def vec_dot(*vecs):
    """
    Dot product across multiple vectors.
    If more than 2 vectors are provided, it computes:
      dot(v1, v2) + dot(v3, v4) + ...
    If odd number of vectors, last one is ignored.
    """
    if len(vecs) < 2:
        raise ValueError('vec_dot requires at least two vectors')

    def dot2(a, b):
        if NP:
            return float(np.dot(a, b))
        return sum((x * y for x, y in zip(a, b)))
    total = 0.0
    for i in range(0, len(vecs) - 1, 2):
        total += dot2(vecs[i], vecs[i + 1])
    return total

def vec_cos_sim(a, b):
    scale = (vec_norm(a) + vec_norm(b)) * 0.1 + 1e-21
    na = vec_norm(a) + scale
    nb = vec_norm(b) + scale
    return vec_dot(a, b) / (na * nb)
AFFORDANCE_BIAS = vec_zero(DIM)
AFFORDANCE_DECAY = 0.991
AFFORDANCE_LR = 0.02

def benchmark_vector_ops():
    start = time.time()
    test_result = vec_add(C, T)
    for _ in range(1000):
        vec_add(C, T)
        vec_norm(test_result)
        vec_dot(S, B)
        vec_cos_sim(C, T)
    return time.time() - start

def logic_mutate(code, C, T, S, B):
    import ast, hashlib
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return code
    h = hashlib.md5(str((vec_norm(C), vec_norm(T), vec_dot(S, B))).encode()).digest()
    a, b = (h[0] % len(funcs), h[1] % len(funcs))
    f1, f2 = (funcs[a], funcs[b])
    new_name = f'auto_func_{h[2] * 256 + h[3]}'
    new_func = ast.parse(f'\ndef {new_name}(C, T, M, S, B):\n    x = {f1.name}(C, T, M, S, B)\n    y = {f2.name}(C, T, M, S, B)\n    return vec_add(x, y) if x is not None and y is not None else C\n').body[0]
    tree.body.append(new_func)
    return ast.unparse(tree)

def self_evolve(current_file, B, S):
    global C, T, M, O, E, collapse_events
    global global_steps
    global last_fitness_delta
    last_fitness_delta = 0.0
    print('[Entered self_evolve]')
    print('[DEBUG-SE] 0: Entered self_evolve, file:', current_file)
    try:
        print('[DEBUG-SE] 1: Opening file for read')
        with open(current_file, 'r', encoding='utf-8', errors='replace') as f:
            print('[DEBUG-SE] 2: File opened, reading lines')
            lines = f.readlines()
            print('[MUT DEBUG 1] Original lines count:', len(lines))
            print('[DEBUG-SE] 4: Computing scores')
        scores = [abs(B[i % len(B)]) for i in range(len(lines))]
        print('[DEBUG-SE] 5: Sorting lines by score')
        scored_lines = sorted(zip(scores, range(len(lines)), lines), key=lambda x: x[0], reverse=True)
        print('[DEBUG-SE] 6: Structural logic mutation')
        parent_code = ''.join(lines)
        child_code = logic_mutate(parent_code, C, T, S, B)
        lines = [ln + '\n' for ln in child_code.splitlines()]
        b_norm = vec_norm(B)
        if b_norm > 0.5:
            change_type = int((abs(S[0]) + abs(S[1]) * 0.5 + abs(S[2]) * 0.3) * 6) % 6
            if change_type == 0 and len(lines) < 1000:
                new_func = ['def log_emergent(msg):\n', "    print(f'Emergent log: {msg}')\n"]
                insert_idx = int(abs(S[1]) * len(lines)) % len(lines)
                if not any(('def log_emergent' in line for line in lines)):
                    lines = lines[:insert_idx] + new_func + lines[insert_idx:]
                    log_evolution_attempt('ADD_FUNCTION', f'Inserted at {insert_idx}:\n' + ''.join(new_func))
            elif change_type == 1 and len(lines) > 10:
                remove_idx = int(abs(S[1]) * len(lines)) % len(lines)
                if 'print' in lines[remove_idx] or '#' in lines[remove_idx]:
                    del lines[remove_idx]
                log_evolution_attempt('REMOVE_LINE', f'idx={remove_idx}')
            elif change_type == 2:
                dupe_idx = int(abs(S[2]) * len(lines)) % len(lines)
                lines.insert(dupe_idx, lines[dupe_idx])
                log_evolution_attempt('DUPLICATE_LINE', f'idx={dupe_idx}')
            elif change_type == 3:
                var_type = int(abs(S[3]) * 3) % 3
                if var_type == 0:
                    new_line = f'EMERGENT_THRESHOLD = {0.85 + S[4] * 0.1:.4f}  # Auto-tuned by phase\n'
                elif var_type == 1:
                    new_line = f'phase_log = []  # Accumulate high-coherence events\n'
                else:
                    new_line = f'STABLE_MODE = True  # Emergent stability flag\n'
                insert_idx = int(abs(S[5 % len(S)]) * len(lines)) % len(lines)
                lines.insert(insert_idx, new_line)
                log_evolution_attempt('ADD_DEFINITION', f'type={var_type}')
            elif change_type == 5:
                current_coh = coherence_scalar(C, T)
                adjustment = int((current_coh - 0.7) * 100000)
                new_steps = global_steps + adjustment
                new_steps = max(1, min(100000, new_steps))
                for k in range(len(lines)):
                    if 'global_steps =' in lines[k]:
                        lines[k] = f'global_steps = {new_steps}  # Auto-adjusted by coherence (coh={current_coh:.3f})\n'
                        print(f'Phase adjusted run length to {new_steps} steps')
                        break
                log_evolution_attempt('ADJUST_global_steps', f'from {global_steps} to {new_steps}')
            elif change_type == 4:
                current_coh = coherence_scalar(C, T)
                new_dim = int(max(8, min(512, GLOBAL_DIM + int((current_coh - 0.7) * 32))))
                for k in range(len(lines)):
                    if lines[k].startswith('GLOBAL_DIM ='):
                        lines[k] = f'GLOBAL_DIM = {new_dim}  # Auto-adjusted by coherence (coh={current_coh:.3f})\n'
                        print(f'Phase adjusted GLOBAL_DIM to {new_dim}')
                        break
                log_evolution_attempt('ADJUST_global_steps', f'from {global_steps} to {new_steps}')
        print('[DEBUG-SE] 7: Writing tmp file')
        for i, line in enumerate(lines):
            if line.strip():
                lines[i] = line.lstrip()
                break
        while lines and lines[0].strip() == '':
            lines.pop(0)
        tmp = current_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8', errors='replace') as f:
            f.writelines(lines)
        print('[DEBUG-SE] 8: Reading tmp_code')
        qfit = quantum_fitness(S[:1])
        env_coverage = 1.0 - vec_cos_sim(S, env_vec)
        old_fitness = fitness(C, T, M, O, E, S, B, collapse_events, vec_norm(C))
        print('[DEBUG-SE] 8.5: old fitness')
        tmp_code = open(tmp, 'r', encoding='utf-8', errors='replace').read()
        child_score = spawn_and_test_child(tmp_code, SANDBOX_DIR, timeout=4)
        qfit_child = quantum_fitness(S[:1])
        env_coverage_child = 1.0 - vec_cos_sim(S, env_vec)
        new_fitness = old_fitness + child_score
        delta = new_fitness - old_fitness
        last_fitness_delta = delta
        record_mutation(tmp_code, delta)
        try:
            print('==== LAST 30 LINES OF MUTATED CODE ====')
            print('\n'.join(tmp_code.splitlines()[-30:]))
            compile(tmp_code, tmp, 'exec')
            with open(DEBUG_LOG_FILE, 'a') as f:
                f.write(f'{time.time():.6f} | COMPILE_OK | fitness_delta={delta:.6f}\n')
                snippet = '\n'.join(tmp_code.splitlines()[-30:])
                with open(SUCCESS_LOG_FILE, 'a') as f:
                    f.write('=== Evolution success log ===\n')
                    f.write(f'{time.time():.6f} | COMPILE_OK | fitness_delta={delta:.6f}\n')
                    f.write('---- attempted code snippet start ----\n')
                    f.write(snippet + '\n')
                    f.write('---- attempted code snippet end ----\n')
                print(f'[SUCCESS LOG FAIL] {e}')
        except Exception as e:
            with open(DEBUG_LOG_FILE, 'a') as f:
                f.write(f'{time.time():.6f} | COMPILE_FAIL | fitness_delta={delta:.6f} | {e}\n')
                f.write('---- attempted code snippet start ----\n')
                snippet = '\n'.join(tmp_code.splitlines()[-30:])
                f.write(snippet + '\n')
                f.write('---- attempted code snippet end ----\n')
            with open(FAIL_LOG_FILE, 'a') as f:
                f.write(f'{time.time():.6f} | COMPILE_FAIL | fitness_delta={delta:.6f} | {e}\n')
                f.write('---- attempted code snippet start ----\n')
                f.write(snippet + '\n')
                f.write('---- attempted code snippet end ----\n')
            raise
        mutated_file = os.path.join(SANDBOX_DIR, f'plic_child_{int(time.time() * 1000)}.py')
        shutil.copy(tmp, mutated_file)
        parent = open(current_file, 'r', encoding='utf-8').read()
        child = open(mutated_file, 'r', encoding='utf-8').read()
        if parent == child:
            print('[WARNING] Child is identical to parent — mutation may have failed')
        else:
            print('[OK] Child differs from parent — mutation applied')
        try:
            log_child_diff(current_file, mutated_file)
        except Exception as e:
            print(f'[DIFF LOG FAIL] {e}')
        print(f'[MUTATION] Wrote mutated version to {mutated_file} (did NOT overwrite parent)')
        print(f'[MUTATION] Wrote mutated version to {mutated_file} (did NOT overwrite parent)')
    except Exception as e:
        print(f'[self_evolve FAILED] {str(e)}')

def self_evolve_det(current_file, B, S):
    global C, T, M, O, E, collapse_events
    import ast, hashlib
    with open(current_file, 'r', encoding='utf-8', errors='replace') as f:
        parent_code = f.read()
    tree = ast.parse(parent_code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) < 2:
        return
    h = hashlib.md5(str((vec_norm(C), vec_norm(T), vec_dot(S, B))).encode()).digest()
    f1 = funcs[h[0] % len(funcs)]
    f2 = funcs[h[1] % len(funcs)]
    name = f"auto_func_{int.from_bytes(h[2:4], 'little')}"
    new_func = ast.parse(f'def {name}(C, T, M, S, B):\n    x = {f1.name}(C, T, M, S, B)\n    y = {f2.name}(C, T, M, S, B)\n    return vec_add(x, y) if x is not None and y is not None else C\n').body[0]
    tree.body.append(new_func)
    child_code = ast.unparse(tree)
    tmp = current_file + '.tmp'
    with open(tmp, 'w', encoding='utf-8', errors='replace') as f:
        f.write(child_code)
    record_mutation(child_code, 0.0)
    print(f'[self_evolve_det] inserted {name}')

def self_meta_evolve(code_text, C, T, S, B, memory=None):
    DIM = len(S)
    blocks = re.split('\\n(?=def |class )', code_text)
    new_blocks = []
    for blk in blocks:
        factor = (vec_norm(C) + vec_dot(S, B) + coherence_scalar(C, T)) * 0.01
        blk = re.sub('(["\\\'])(.*?)(\\d+)(.*?)(["\\\'])', lambda m: m.group(1) + m.group(2) + str(int(float(m.group(3)) * (1 + factor))) + m.group(4) + m.group(5), blk)
        os.makedirs(os.path.dirname(DEBUG_LOG_FILE), exist_ok=True)
        with open(DEBUG_LOG_FILE, 'a', encoding='utf-8', errors='replace') as f:
            f.write(f"{time.time():.6f} | {blk[:100].replace(chr(10), ' ')}\n")
        new_blocks.append(blk)
    new_code = '\n'.join(new_blocks)
    if memory is None:
        memory = memory or {hashlib.md5(open(os.path.join(EVOLUTION_DIR, f), 'rb').read()).hexdigest(): code_to_semantic_vector(open(os.path.join(EVOLUTION_DIR, f), 'r', encoding='utf-8', errors='replace').read(), DIM) for f in os.listdir(EVOLUTION_DIR) if f.endswith('.py')}
    memory[hashlib.md5(new_code.encode('utf-8')).hexdigest()] = code_to_semantic_vector(new_code, DIM)
    print(f'[DEBUG] meta_memory size (disk+run): {len(memory)}')
    return (new_code, memory)
_token_re = re.compile('[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|->|=>|[\\+\\-\\*\\/%=<>]|\\d+|\\"[^\\"]*\\"|\'[^\']*\'')

def load_vector(path, fallback_scale=0.02):
    if NP and os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass
    if not NP and os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return json.load(f)
        except Exception:
            pass
    return vec_rand(DIM, scale=fallback_scale)

def save_vector(path, v):
    try:
        if NP:
            np.save(path, v)
        else:
            with open(path, 'w') as f:
                json.dump(v, f)
    except Exception:
        pass
B_FILE = os.path.join(SANDBOX_DIR, 'B_vector.npy' if NP else 'B_vector.json')
B = load_vector(B_FILE)
print('Loaded B vector:', B[:5], '...')
S = load_vector(SEMANTIC_FILE)
if not S or len(S) != DIM:
    S = vec_rand(DIM, scale=0.01)
print('Loaded S vector:', S[:5], '...')
C = load_vector(C_FILE, fallback_scale=0.05)
print('Loaded C vector:', C[:5], '...')
T = load_vector(CONTINUITY_FILE)
print('Loaded T vector:', T[:5], '...')
W_NN = load_vector(os.path.join(SANDBOX_DIR, 'W_NN.json'), fallback_scale=0.02)
if len(W_NN) != DIM * DIM:
    W_NN = vec_rand(DIM * DIM, 0.02)
print('Loaded W_NN:', W_NN[:5], '...')
b_NN = load_vector(os.path.join(SANDBOX_DIR, 'b_NN.json'), fallback_scale=0.01)
if len(b_NN) != DIM:
    b_NN = vec_zero(DIM)
print('Loaded b_NN:', b_NN[:5], '...')
W_LANG = load_vector(os.path.join(SANDBOX_DIR, 'W_LANG.json'), fallback_scale=0.02)
if len(W_LANG) != DIM:
    W_LANG = vec_rand(DIM, 0.02)
print('Loaded W_LANG:', W_LANG[:5], '...')
b_LANG = load_vector(os.path.join(SANDBOX_DIR, 'b_LANG.json'), fallback_scale=0.01)
if len(b_LANG) != DIM:
    b_LANG = vec_rand(DIM, 0.01)
print('Loaded b_LANG:', b_LANG[:5], '...')

def code_to_semantic_vector(code_text, dim=DIM):
    tokens = _token_re.findall(code_text)
    counts = [0.0] * dim
    for tok in tokens:
        h = hashlib.md5(tok.encode('utf-8')).digest()
        idx = int.from_bytes(h[:2], 'little') % dim
        counts[idx] += 1.0
    if NP:
        v = np.array(counts, dtype=float)
        norm = np.linalg.norm(v) + 1e-09
        return v / norm
    else:
        n = math.sqrt(sum((x * x for x in counts))) + 1e-09
        return [x / n for x in counts]

def bytes_to_vector(path, dim=256):
    with open(path, 'rb') as f:
        data = f.read()
    v = [0.0] * dim
    for b in data:
        v[b % dim] += 1.0
    norm = math.sqrt(sum((x * x for x in v))) + 1e-09
    return [x / norm for x in v]
print(bytes_to_vector(__file__)[:10])

def byte_ngrams(data, n=4, dim=1024):
    v = [0.0] * dim
    for i in range(len(data) - n):
        h = hashlib.md5(data[i:i + n]).digest()
        idx = int.from_bytes(h[:2], 'little') % dim
        v[idx] += 1
    return normalize(v)

def binary_structural_features(data):
    features = {'entropy': shannon_entropy(data), 'zero_ratio': data.count(0) / len(data), 'printable_ratio': sum((32 <= b < 127 for b in data)) / len(data), 'repetition': repetition_score(data), 'block_entropy': block_entropy(data, block=512)}

def file_to_field_vector(path):
    data = open(path, 'rb').read()
    v1 = bytes_to_vector(path, dim=DIM // 2)
    v2 = byte_ngrams(data, n=4, dim=DIM - DIM // 2)
    return normalize(v1 + v2)

def read_all_files(base_dir):
    code_text = ''
    for root, _, files in os.walk(base_dir):
        for filename in files:
            path = os.path.join(root, filename)
            print('Reading file:', path)
            try:
                if not should_read(path):
                    continue
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    code_text += '\n' + f.read()
            except Exception:
                pass
    return code_text

def read_code_from_sandbox(sandbox_dir):
    path = os.path.join(sandbox_dir, 'plic_ai_v4_1.py')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception:
            pass
    try:
        here = __file__
        with open(here, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception:
        return ''
M = vec_rand(DIM, scale=0.03)
O = vec_rand(DIM, scale=0.06)
tn = vec_norm(T) + 1e-09
if tn != 0:
    T = vec_mul_scalar(T, 0.8 / tn)
SEM_LEARN_RATE = vec_norm(S) * 0.1 + 0.005
E = vec_rand(DIM, scale=EXPLORE_BASE)
print('Loaded E vector:', S[:5], '...')
print('[DEBUG VEC] Test add:', vec_add([1, 2], [3, 4])[:2])
print('[DEBUG VEC] Test norm:', vec_norm([3, 4]))
print('[DEBUG VEC] Test cos_sim:', vec_cos_sim([1, 0], [0, 1]))
print('[DEBUG VEC] Test rand sample:', vec_rand(5, scale=0.1))
collapse_events = 0
B_FILE = os.path.join(SANDBOX_DIR, 'B_vector.json')
lr_NN = 0.005
selfS = code_to_semantic_vector(read_code_from_sandbox(SANDBOX_DIR))
S = vec_add(vec_mul_scalar(S, 0.95), vec_mul_scalar(selfS, 0.05))
S = vec_mul_scalar(S, 1.0 / (vec_norm(S) + 1e-09))
CORPUS_FILE = os.path.join(SANDBOX_DIR, 'corpus.txt')
try:
    with open(CORPUS_FILE, 'r', encoding='utf-8', errors='replace') as f:
        corpus = json.load(f)
except Exception:
    corpus = {}
text = read_all_files(SANDBOX_DIR)
for tok in _token_re.findall(text):
    reinforce = 1.0 + max(0.0, last_fitness_delta)
    corpus[tok] = corpus.get(tok, 0.0) + reinforce
DECAY_FACTOR = 0.07
for tok in list(corpus.keys()):
    corpus[tok] *= DECAY_FACTOR
    if corpus[tok] < 0.01:
        del corpus[tok]
with open(CORPUS_FILE, 'w', encoding='utf-8', errors='replace') as f:
    json.dump(corpus, f)
MAX_CORPUS_SIZE_BYTES = 50 * 1024 * 1024
current_size = os.path.getsize(CORPUS_FILE)
if current_size > MAX_CORPUS_SIZE_BYTES:
    print(f'[CORPUS CAP] File is {current_size / 1024 / 1024:.1f} MB — trimming to ~50 MB...')
    with open(CORPUS_FILE, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    sorted_tokens = sorted(data.items(), key=lambda item: item[1], reverse=True)
    avg_entry_size = current_size / len(data) if data else 200
    keep_count = int(MAX_CORPUS_SIZE_BYTES / avg_entry_size)
    keep_count = max(50000, min(keep_count, len(sorted_tokens)))
    trimmed_corpus = dict(sorted_tokens[:keep_count])
    with open(CORPUS_FILE, 'w', encoding='utf-8', errors='replace') as f:
        json.dump(trimmed_corpus, f)
    corpus.clear()
    corpus.update(trimmed_corpus)
    print(f'[CORPUS TRIMMED] Now {len(corpus)} tokens, ~{os.path.getsize(CORPUS_FILE) / 1024 / 1024:.1f} MB')
corpus_list = list(corpus.keys())
print(f'Corpus ready, {len(corpus_list)} tokens.')
last_phrase_path = os.path.join(SANDBOX_DIR, 'last_phrase.txt')
if os.path.exists(last_phrase_path):
    with open(last_phrase_path, 'r', encoding='utf-8', errors='replace') as f:
        for tok in _token_re.findall(f.read()):
            corpus[tok] = corpus.get(tok, 0.0) + 1.0
x = [0.0] * DIM
tokens = list(corpus.keys())[:DIM]
for i, tok in enumerate(tokens):
    x[i] += corpus[tok]

def coherence_scalar(C_field, T_field):
    diff = vec_sub(T_field, C_field)
    return 1.0 / (1.0 + vec_norm(diff))

def pressure_scalar(C_field, M_field):
    return (vec_norm(C_field) + 0.5 * vec_norm(M_field)) / (1.0 + vec_norm(C_field) + vec_norm(M_field))

def resistance_scalar(T_field, O_field):
    return (vec_norm(T_field) + vec_norm(O_field)) / (1.0 + vec_norm(T_field) + vec_norm(O_field))
mutation_memory = {}

def fitness(C, T, M, O, E, S, B, collapse_events, entropy, qfit=0.0):
    coh = coherence_scalar(C, T)
    pres = pressure_scalar(C, M)
    res = resistance_scalar(T, O)
    base = coh * 0.6 + pres * 0.2 + (1 - res) * 0.1 - 0.1 * entropy - 0.2 * math.log1p(collapse_events)
    return base + 0.1 * qfit + 0.1 * env_coverage

def record_mutation(code_text, delta):
    h = hashlib.md5(code_text.encode('utf-8', errors='replace')).hexdigest()
    mutation_memory[h] = delta

def collapse_condition(pressure, resistance, threshold=0.08):
    return pressure - resistance > threshold

def update_fields(C, T, M, O, E, S, self_org_rate=SELF_ORG_RATE, noise_scale=NOISE_SCALE, explore_strength=0.0, sem_learn=SEM_LEARN_RATE):
    noise = vec_mul_scalar(S, 0.01)
    env_stress = get_hardware_stress(DIM)
    M = vec_add(M, vec_mul_scalar(env_stress, 0.005))
    noise_scale += vec_norm(env_stress) * 0.001
    drift = vec_mul_scalar(M, 0.02)
    C = vec_add(C, vec_add(noise, vec_mul_scalar(drift, -1.0)))
    C = vec_add(C, vec_mul_scalar(E, explore_strength))
    pressure = pressure_scalar(C, M)
    resistance = resistance_scalar(T, O)
    directional = vec_add(C, M)
    mag = vec_norm(directional) + 1e-09
    if pressure - resistance > 0.0:
        T = vec_add(T, vec_mul_scalar(directional, -0.1 * self_org_rate / mag))
    M = vec_add(M, vec_mul_scalar(vec_sub(O, C), 0.05 * self_org_rate))
    if pressure > resistance:
        O = vec_add(O, vec_mul_scalar(vec_mul_scalar(T, -1.0), 0.02 * self_org_rate))
    else:
        O = vec_add(O, vec_mul_scalar(T, 0.04 * self_org_rate))
    O = vec_add(O, vec_rand(DIM, scale=noise_scale * 0.1))
    if S is not None:
        semantic_influence = vec_mul_scalar(S, sem_learn * 0.5)
        T = vec_add(T, semantic_influence)
        if coherence_scalar(C, T) > 0.9:
            proposed_T = vec_add(T, vec_mul_scalar(S, sem_learn))
            T = proposed_T
    learning = vec_mul_scalar(vec_add(vec_mul_scalar(C, 0.2), vec_mul_scalar(O, 0.8)), 0.005 * self_org_rate)
    T = vec_add(T, learning)
    C = vec_clip(C, -10, 10)
    T = vec_clip(T, -10, 10)
    M = vec_clip(M, -10, 10)
    O = vec_clip(O, -10, 10)
    return (C, T, M, O)

def handle_collapse(C, T, M, O):
    directional = vec_add(C, M)
    compress_amount = 0.05
    T = vec_add(T, vec_mul_scalar(directional, compress_amount))
    C = vec_mul_scalar(C, 0.5)
    O = vec_mul_scalar(O, 0.9)
    return (C, T, M, O)

def log_step(step, psm, pressure, resistance, collapse_flag, pred_err, entropy, explore_mag):
    entry = {'step': int(step), 'psm': float(psm), 'pressure': float(pressure), 'resistance': float(resistance), 'collapse': bool(collapse_flag), 'pred_err': None if pred_err is None else float(pred_err), 'entropy': float(entropy), 'explore_mag': float(explore_mag), 'timestamp': time.time()}
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass

def build_semantic_vector_from_code(sandbox_dir):
    code = read_code_from_sandbox(sandbox_dir)
    if not code:
        return None
    return code_to_semantic_vector(code, dim=DIM)

def code_to_semantic_vector(code_text, dim=DIM):
    tokens = _token_re.findall(code_text)
    counts = [0.0] * dim
    for tok in tokens:
        h = hashlib.md5(tok.encode('utf-8')).digest()
        idx = int.from_bytes(h[:2], 'little') % dim
        counts[idx] += 1.0
    if NP:
        v = np.array(counts, dtype=float)
        norm = np.linalg.norm(v) + 1e-09
        return v / norm
    else:
        n = math.sqrt(sum((x * x for x in counts))) + 1e-09
        return [x / n for x in counts]

def phase(v):
    n = vec_norm(v)
    return None if n == 0 else [vi / n for vi in v]

def phase_lift(v, c=1.1):
    r = vec_norm(v)
    return [vi * r ** (c - 1) for vi in v]

def plic_add(x, y):
    px, py = (phase(x), phase(y))
    if px is None:
        return y[:]
    if py is None:
        return x[:]
    dot = sum((px[i] * py[i] for i in range(len(px))))
    return phase_lift(x) if abs(dot - 1.0) < 1e-06 else vec_add(x, y)

def tiny_lang_nn(S, corpus_list, DIM=DIM, H=64, lr=5e-05):
    print('[TinyNN] CALLED')
    Wf, bf = (os.path.join(SANDBOX_DIR, 'W_LANG.json'), os.path.join(SANDBOX_DIR, 'b_LANG.json'))
    try:
        W = json.load(open(Wf))
        W += [0.0] * (H * DIM - len(W))
        W = W[:H * DIM]
        print('[TinyNN] W loaded')
    except:
        W = [0.0] * (H * DIM)
        print('[TinyNN] W initialized')
    try:
        b = json.load(open(bf))
        print('[TinyNN] b loaded')
    except:
        b = [0.0] * H
        print('[TinyNN] b initialized')
    if len(W) != H * DIM or len(b) != H:
        print('[TinyNN] LANG size mismatch, skipping update')
        print('  W_LANG shape:', getattr(W_LANG, 'shape', None))
        print('  b_LANG shape:', getattr(b_LANG, 'shape', None))
        return S
    x = [0.0] * DIM
    tokens = list(corpus.keys())[:DIM]
    for i, tok in enumerate(tokens):
        weight = corpus[tok]
        x[i] += weight
    h_raw = [sum((W[i * DIM + j] * x[j] for j in range(DIM))) + b[i] for i in range(H)]
    h = plic_add(h_raw, x[:H])
    for i in range(H):
        w_i = W[i * DIM:(i + 1) * DIM]
        delta = vec_mul_scalar(x, h[i])
        w_new = plic_add(w_i, delta)
        W[i * DIM:(i + 1) * DIM] = w_new
    S = plic_add(S, vec_mul_scalar(h_raw, 0.001))
    try:
        json.dump(W, open(Wf, 'w'))
        json.dump(b, open(bf, 'w'))
    except:
        print('[TinyNN] save failed')
    return S
S = vec_add(vec_mul_scalar(S, 0.95), vec_mul_scalar(selfS, 0.05))
S = vec_mul_scalar(S, 1.0 / (vec_norm(S) + 1e-09))
def auto_func_37911(C, T, M, S, B):
    x = save_vector(C, T, M, S, B)
    y = build_semantic_vector_from_code(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_13080(C, T, M, S, B):
    x = vec_copy(C, T, M, S, B)
    y = record_mutation(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C
def auto_func_5098(C, T, M, S, B):
    x = vec_to_str(C, T, M, S, B)
    y = auto_func_89(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_46046(C, T, M, S, B):
    x = auto_func_10132(C, T, M, S, B)
    y = vec_add(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_35492(C, T, M, S, B):
    x = vec_sub(C, T, M, S, B)
    y = auto_func_91(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_35667(C, T, M, S, B):
    x = auto_func_10132(C, T, M, S, B)
    y = auto_func_10132(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_10132(C, T, M, S, B):
    x = auto_func_91(C, T, M, S, B)
    y = auto_func_13374(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_13374(C, T, M, S, B):
    x = vec_add(C, T, M, S, B)
    y = vec_add(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_27173(C, T, M, S, B):
    x = auto_func_91(C, T, M, S, B)
    y = vec_add(C, T, M, S, B)
    return vec_add(x, y) if x is not None and y is not None else C

def auto_func_91(C, T, M, S, B):
    C = vec_add(C, C)
    C = vec_add(C, T)
    C = vec_add(C, M)
    return C
    C = vec_add(C, T)

def auto_func_89(C, T, M, S, B):
    C = vec_add(C, C)
    C = vec_add(C, T)
    C = vec_add(C, M)
    return C
    C = vec_add(C, T)
    C = vec_add(C, M)
    return C
    C = vec_add(C, M)
    return C

def run():
    global W_TEXT, b_TEXT
    global spawn_and_test_child
    global EXPLORE_BASE, EXPLORE_DECAY
    global C, T, M, O, E, S, B, collapse_events, SEM_LEARN_RATE, SEM_PERSIST_THRESH, ENTROPY_CEILING
    W_TEXT = W_TEXT if 'W_TEXT' in globals() and len(W_TEXT) == DIM * DIM else [0.0] * (DIM * DIM)
    b_TEXT = b_TEXT if 'b_TEXT' in globals() and len(b_TEXT) == DIM else [0.0] * DIM
    print('PLIC-AI v4.1 (Structural Semantics + ASPC)')
    try:
        with open(__file__, 'r', encoding='utf-8', errors='replace') as f:
            line_count = len(f.readlines())
        print(f'[SIZE] Current source code lines: {line_count}')
    except Exception as e:
        line_count = -1
        print(f'[SIZE] Could not read line count: {e}')
    print('Sandbox dir:', SANDBOX_DIR)
    start = time.time()
    drive_root = os.path.abspath(__file__)
    env_vec = env_sonar_ping(drive_root)
    newS = build_semantic_vector_from_code(SANDBOX_DIR)
    H_BIAS = get_hardware_stress(DIM)
    token_weights = list(zip(corpus_list[:DIM], W_TEXT[:DIM]))
    top_tokens = sorted(token_weights, key=lambda t: t[1], reverse=True)[:10]
    print('TinyNN top learned human tokens:', top_tokens)
    S = vec_add(S, vec_mul_scalar(H_BIAS, 0.05))
    if newS is not None:
        S = vec_add(vec_mul_scalar(S, 1.0 - 0.05), vec_mul_scalar(newS, 0.05))
        sn = vec_norm(S) + 1e-09
        S = vec_mul_scalar(S, 1.0 / sn)
    file_vec = file_to_field_vector(__file__)
    S = vec_add(vec_mul_scalar(S, 0.9), vec_mul_scalar(file_vec, 0.1))
    baseline_entropy = vec_norm(C) + 1e-09
    target_entropy = baseline_entropy * 1.25
    explore_strength = EXPLORE_BASE
    for step in range(global_steps):
        print(f'[STEP {step + 1}/{global_steps}]')
        ent = vec_norm(C)
        if step == 0:
            print(f'[EVOLVE] global_steps in effect this run = {global_steps}')
        if step == 0:
            print(f'[EVOLVE] GLOBAL_DIM in effect this run = {DIM}')
        if step == 0:
            print('Tiny NN hidden sample:', W_NN[:3], 'b sample:', b_NN[:3])
            S = tiny_lang_nn(S, corpus_list)
            human_text = read_all_files(SANDBOX_DIR)
            W_TEXT = load_vector(os.path.join(SANDBOX_DIR, 'W_TEXT.json'), fallback_scale=0.02)
            W_TEXT += [0.0] * (DIM * DIM - len(W_TEXT))
            b_TEXT = load_vector(os.path.join(SANDBOX_DIR, 'b_TEXT.json'), fallback_scale=0.2)
            b_TEXT += [0.0] * (DIM - len(b_TEXT))
            tokens_txt = _token_re.findall(human_text)
            x_txt = [0.0] * DIM
        for i, _ in enumerate(tokens_txt):
            x_txt[i % DIM] += 1
        h_txt = [math.tanh(sum((W_TEXT[i * DIM + j] * x_txt[j] for j in range(DIM))) + b_TEXT[i]) for i in range(DIM)]
        S = vec_add(S, vec_mul_scalar(x_txt, 0.001))
        lr_txt = 0.0001
        for i in range(DIM):
            for j in range(DIM):
                W_TEXT[i * DIM + j] += lr_txt * h_txt[i] * x_txt[j]
            b_TEXT[i] += lr_txt * h_txt[i]
        json.dump(W_TEXT, open(os.path.join(SANDBOX_DIR, 'W_TEXT.json'), 'w'))
        json.dump(b_TEXT, open(os.path.join(SANDBOX_DIR, 'b_TEXT.json'), 'w'))
        if 'meta_memory' not in globals():
            meta_memory = {}
            print('meta_memory initialized')
        if ent < target_entropy:
            explore_dir = vec_sub(S, T)
            if vec_norm(explore_dir) < 0.01:
                explore_dir = vec_mul_scalar(B, 0.1)
            anti_thread = vec_mul_scalar(explore_dir, 0.5)
            E = vec_add(vec_mul_scalar(E, 0.9), vec_mul_scalar(anti_thread, 0.1))
            en = vec_norm(E) + 1e-09
            if en > EXPLORE_MAX:
                E = vec_mul_scalar(E, EXPLORE_MAX / en)
            explore_strength = min(EXPLORE_MAX, explore_strength * 1.02 + 0.001)
        else:
            E = vec_mul_scalar(E, 0.995)
            explore_strength *= EXPLORE_DECAY
            explore_strength = max(EXPLORE_BASE * 0.1, explore_strength)
        C_old = vec_copy(C)

        C, T, M, O = update_fields(C, T, M, O, E, S,
                                   self_org_rate=SELF_ORG_RATE,
                                   noise_scale=NOISE_SCALE,
                                   explore_strength=explore_strength,
                                   sem_learn=SEM_LEARN_RATE)

        err_vec = vec_sub(C, C_old)
        prediction_error = vec_norm(err_vec)

        for i in range(len(W_NN)):
            W_NN[i] += 0.001 * (S[i % DIM] - W_NN[i])
        print(f"[LEARN] step={step} prediction_error={prediction_error:.6f} W_NN sample={W_NN[:5]}")

        new_code, meta_memory = self_meta_evolve(read_code_from_sandbox(SANDBOX_DIR), C, T, S, B, meta_memory)
        old_lines = read_code_from_sandbox(SANDBOX_DIR).splitlines()
        new_lines = new_code.splitlines()
        diff_lines = [line for old, line in zip(old_lines, new_lines) if old != line]
        print('[DEBUG] changed lines:', diff_lines[:5])
        for name, fn in list(globals().items()):
            if name.startswith('auto_func_'):
                try:
                    result = fn(C, T, M, S, B)
                    if result is not None and len(result) == DIM and all((isinstance(x, (int, float)) for x in result)):
                        print(f'[USEFUL] {name} returned valid vector (norm={vec_norm(result):.4f})')
                        C = vec_clip(result, -10, 10)
                    elif result is not None:
                        print(f"[IGNORED] {name} returned invalid: type={type(result)}, len={(len(result) if hasattr(result, '__len__') else 'no len')}")
                    else:
                        print(f'[IGNORED] {name} returned None')
                except Exception as e:
                    print(f'[SAFE-CALL] {name} failed: {str(e)[:80]}...')
        if 'A' not in globals():
            A = vec_zero(DIM)
        A = vec_add(A, vec_mul_scalar(vec_sub(T, C), 0.001))
        A = vec_clip(A, -0.05, 0.05)
        SEM_LEARN_RATE = SEM_LEARN_MIN + (SEM_LEARN_MAX - SEM_LEARN_MIN) * abs(A[0])
        EXPLORE_BASE = max(0.001, EXPLORE_BASE * (1.0 + A[1]))
        EXPLORE_DECAY = max(0.9, min(0.999, EXPLORE_DECAY * (1.0 + A[2])))
        if vec_norm(A) > 0.3 and coherence_scalar(C, T) > 0.92:
            T = vec_add(T, vec_mul_scalar(S, SEM_LEARN_RATE * 1.5))
            print('Autonomous consolidation: phase deepened')
        if step % 500 == 0:
            print(f'[AUTO] SEM_LEARN_RATE={SEM_LEARN_RATE:.5f}, EXPLORE_BASE={EXPLORE_BASE:.5f}, EXPLORE_DECAY={EXPLORE_DECAY:.5f}, |A|={vec_norm(A):.4f}')
        B = vec_add(B, vec_mul_scalar(vec_sub(C, T), 0.02))
        B = vec_clip(B, -1, 1)
        thought_vec = vec_add(vec_mul_scalar(S, 0.5), vec_mul_scalar(C, 0.3))
        thought_vec = vec_add(thought_vec, vec_mul_scalar(T, 0.2))
        thought_vec = vec_add(thought_vec, vec_mul_scalar(B, 0.1))
        phrase = ' '.join((corpus_list[int(abs(thought_vec[i % len(thought_vec)]) * len(corpus_list)) % len(corpus_list)] for i in range(20)))
        print('B vector (sample):', B[:5])
        print('Thought vector (sample):', thought_vec[:5])
        print('Emergent phrase:', phrase)
        phrase_file = os.path.join(SANDBOX_DIR, 'last_phrase.txt')
        with open(phrase_file, 'a', encoding='utf-8', errors='replace') as f:
            f.write(phrase)
        coh = coherence_scalar(C, T)
        pressure = pressure_scalar(C, M)
        resistance = resistance_scalar(T, O)
        psm = (coh + resistance) / 2.0
        entropy = vec_norm(C)
        pred_err = None
        collapsed = collapse_condition(pressure, resistance, threshold=0.08)
        if collapsed:
            collapse_events += 1
            C, T, M, O = handle_collapse(C, T, M, O)
            explore_strength *= 0.7
            print(f'[DEBUG] Collapse #{collapse_events} handled. Pressure={pressure:.3f}, Resistance={resistance:.3f}')
        h = vec_norm(C) / (1 + vec_norm(C))
        for i in range(DIM):
            b_NN[i] += lr_NN * (S[i] - h)
            for j in range(DIM):
                W_NN[i * DIM + j] += lr_NN * (S[j] - h) * S[j]

        def spawn_and_test_child(code_text, sandbox, global_steps_required=500, timeout=10):
            print('[CHILD DEBUG] Child code length:', len(child_code))
            ts = int(time.time())
            child = os.path.join(sandbox, f'plic_child_{ts}.py')
            with open(child, 'w', encoding='utf-8', errors='replace') as f:
                f.write(code_text)
            ok = False
            try:
                p = subprocess.run(['python', child], timeout=timeout, capture_output=True, text=True, encoding='utf-8', errors='replace')
                ok = p.returncode == 0
                with open(os.path.join(sandbox, 'evolution_log.txt'), 'a') as log:
                    log.write(f"{child} :: {('SURVIVED' if ok else 'DIED')}\n")
                return 1.0 if ok else -10.0
            except Exception as e:
                with open(os.path.join(sandbox, 'evolution_log.txt'), 'a') as log:
                    log.write(f'{child} :: CRASH {e}\n')
                return 1.0 if ok else -1.0
        self_evolve(__file__, B, S)
        self_evolve_det(__file__, B, S)
        print('[EVOLVE] Both self_evolve routines executed')
        if psm > 0.85 and (pred_err is None or (pred_err is not None and pred_err < 0.01)):
            SEM_LEARN_RATE *= 1.02
            SEM_PERSIST_THRESH *= 0.98
        if psm < 0.55 or (pred_err is not None and pred_err > 0.05):
            SEM_LEARN_RATE *= 0.9
            explore_strength *= 1.05
        if resistance > 0.95:
            ENTROPY_CEILING = min(ENTROPY_CEILING + 0.5, ENTROPY_CEILING_MAX)
        elif pressure > resistance:
            ENTROPY_CEILING = max(ENTROPY_CEILING - 0.5, ENTROPY_CEILING_MIN)
        SEM_LEARN_RATE = max(SEM_LEARN_MIN, min(SEM_LEARN_RATE, SEM_LEARN_MAX))
        SEM_PERSIST_THRESH = max(SEM_THRESH_MIN, min(SEM_PERSIST_THRESH, SEM_THRESH_MAX))
        ENTROPY_CEILING = max(ENTROPY_CEILING_MIN, min(ENTROPY_CEILING, ENTROPY_CEILING_MAX))
        if step % 500 == 0 and step > 0:
            coh_before = coherence_scalar(C, T)
            T_test = vec_add(T, vec_mul_scalar(S, 0.5 * SEM_LEARN_RATE))
            coh_after = coherence_scalar(C, T_test)
            consolidate_gain = coh_after - coh_before
            if consolidate_gain > SEM_PERSIST_THRESH:
                T = T_test
                save_vector(SEMANTIC_FILE, S)
                save_vector(CONTINUITY_FILE, T)
                print(f'Emergent consolidation at step {step}: gain = {consolidate_gain:.6f}')
        log_step(step, psm, pressure, resistance, collapsed, pred_err, entropy, explore_strength)
        if step % 300 == 0:
            pred_display = 'None' if pred_err is None else f'{pred_err:.6f}'
            print(f'step={step} psm={psm:.3f} pressure={pressure:.3f} resistance={resistance:.3f} entropy={entropy:.4f} SEM_learn={SEM_LEARN_RATE:.4f} SEM_thresh={SEM_PERSIST_THRESH:.6f} collapses={collapse_events}')
    try:
        save_vector(B_FILE, B)
        print(f'Saved B OK: {B_FILE} (norm={vec_norm(B):.4f})')
    except Exception as e:
        print(f'Save B vector FAILED: {B_FILE} - {e}')
    save_vector(CONTINUITY_FILE, T)
    print(f'Saved T OK: {CONTINUITY_FILE} (norm={vec_norm(T):.4f})')
    save_vector(SEMANTIC_FILE, S)
    print(f'Saved S OK: {SEMANTIC_FILE} (norm={vec_norm(S):.4f})')
    elapsed = time.time() - start
    print('Run complete. Time: {:.2f}s  Collapses: {}'.format(elapsed, collapse_events))
    try:
        with open(RUNS_SUMMARY, 'a') as f:
            f.write(json.dumps({'time': time.time(), 'global_steps': global_steps, 'collapses': int(collapse_events), 'final_psm': float(psm)}) + '\n')
    except Exception:
        pass
    with open(__file__, 'r', encoding='utf-8', errors='replace') as f:
        parent_code = f.read()
        print('[CHILD DEBUG 1.0] Parent code length:', len(parent_code))
    child_code = logic_mutate(parent_code, C, T, S, B)
    parent_len = len(parent_code)
    child_score = spawn_and_test_child(child_code, SANDBOX_DIR) + max(0, (parent_len - len(child_code)) // 1e-05)
    print(f'[-CHILD-] len {len(child_code)} vs {parent_len} → bonus {max(0, (parent_len - len(child_code)) * 0.0005):+.5f}')
    aff = spawn_and_test_child(parent_code, SANDBOX_DIR)
    print(f'[-CHILD-] len {len(child_code)} vs {parent_len}')
    if aff > 0:
        print('[CHILD] Survived!')
    else:
        print('[CHILD] Died / timed out')
        save_vector(f, W_LANG)
        save_vector(f, b_LANG)
        print('[TinyNN] Weights inherited:', f'W_LANG sample={W_LANG[:5]}', f'b_LANG sample={b_LANG[:5]}', f'|W|={vec_norm(W_LANG):.4f}', f'|b|={vec_norm(b_LANG):.4f}')
    global AFFORDANCE_BIAS
    AFFORDANCE_BIAS = vec_mul_scalar(AFFORDANCE_BIAS, AFFORDANCE_DECAY)
    code_vec = code_to_semantic_vector(parent_code, DIM)
    AFFORDANCE_BIAS = vec_add(AFFORDANCE_BIAS, vec_mul_scalar(code_vec, AFFORDANCE_LR * aff))
    save_vector(os.path.join(SANDBOX_DIR, 'W_NN.json'), W_NN)
    print('Saved W_NN vector.')
    save_vector(os.path.join(SANDBOX_DIR, 'b_NN.json'), b_NN)
    print('Saved b_NN vector.')
    save_vector(C_FILE, C)
    print('Saved C vector to disk.')
    save_mutation_memory()
    print('Saved MuTation MeMory.')
    json.dump(W_LANG, open(os.path.join(SANDBOX_DIR, 'W_LANG.json'), 'w'))
    print('saved W_LANG')
    json.dump(b_LANG, open(os.path.join(SANDBOX_DIR, 'b_LANG.json'), 'w'))
    print('saved b_LANG')
    child_files = sorted([f for f in os.listdir(SANDBOX_DIR) if f.startswith('plic_child_') and f.endswith('.py')], key=lambda f: os.path.getctime(os.path.join(SANDBOX_DIR, f)), reverse=True)
    for f in child_files[10:]:
        os.remove(os.path.join(SANDBOX_DIR, f))
    print(f'Cleaned old child files; kept last {min(8, len(child_files))} survivors')
if __name__ == '__main__':
    while current_cycle < MAX_CYCLES:
        current_cycle += 1
        print(f'[CYCLE {current_cycle}/{MAX_CYCLES}] Starting full run...')
        run()
        print(f'[CYCLE {current_cycle}/{MAX_CYCLES}] Finished. Mutating for next cycle...')
        time.sleep(2)
    print('All cycles complete.')

